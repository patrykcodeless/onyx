"""Skill-push fan-out reliability behaviors (ext-dep).

The happy-path skill-push contract (public/private landing, disable, grants
flip, bundle replace, delete, dedup) is pinned end-to-end against real pods in
the k8s integration lane (``tests/integration/tests/craft/k8s/test_skill_push``).

This module covers the two behaviors that are awkward or impossible to drive
against a live cluster and belong at the stub layer:

- **State filtering** — ``get_sandbox_user_map`` only returns RUNNING
  sandboxes, so sleeping/terminated pods never receive a write. Driven here by
  putting a sandbox row in the dormant state and asserting the manager's write
  is never invoked.
- **Per-sandbox failure isolation** — one sandbox's ``FatalWriteError`` must
  not abort the fan-out to the others. Driven via ``StubSandboxManager``'s
  fault-injection map, which raises an error we cannot reproduce on demand
  against a real pod.

All run against real Postgres with the sandbox manager stubbed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from uuid import uuid4

import pytest
from sqlalchemy.orm import Session

from onyx.db.enums import SandboxStatus
from onyx.db.models import Skill
from onyx.server.features.build.sandbox.models import FatalWriteError
from onyx.skills.push import push_skills_for_users
from tests.common.craft.stubs import StubSandboxManager
from tests.external_dependency_unit.craft.db_helpers import make_sandbox
from tests.external_dependency_unit.craft.db_helpers import make_user


class TestSkillPushStateFiltering:
    @pytest.mark.parametrize(
        "status",
        [SandboxStatus.SLEEPING, SandboxStatus.TERMINATED],
    )
    def test_push_skips_dormant_sandboxes(
        self,
        db_session: Session,
        seeded_skill: Callable[..., Skill],
        stub_sandbox_manager: StubSandboxManager,
        monkeypatch: pytest.MonkeyPatch,
        status: SandboxStatus,
    ) -> None:
        # A write would succeed silently if it were attempted — so a non-zero
        # count means the dormant sandbox slipped through state filtering.
        stub_sandbox_manager.write_files_to_sandbox_silent = True
        monkeypatch.setattr(
            "onyx.skills.push.get_sandbox_manager",
            lambda: stub_sandbox_manager,
        )

        user = make_user(db_session)
        make_sandbox(db_session, user, status=status)

        seeded_skill(
            slug=f"dormant-{status.value}-{uuid4().hex[:6]}",
            public=True,
            bundle_files={"SKILL.md": "anything\n"},
        )

        # Scope the push to this user: a public skill would otherwise fan out to
        # every RUNNING sandbox in the shared DB (incl. rows other tests left
        # behind). This user's only sandbox is dormant, so get_sandbox_user_map
        # excludes it and nothing is written.
        push_skills_for_users({user.id}, db_session)

        assert stub_sandbox_manager.write_files_to_sandbox_count == 0


class TestSkillPushFailureIsolation:
    def test_one_failing_sandbox_does_not_abort_push_to_others(
        self,
        db_session: Session,
        seeded_skill: Callable[..., Skill],
        failing_sandbox_manager: Callable[..., StubSandboxManager],
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        user_a = make_user(db_session)
        user_b = make_user(db_session)
        user_c = make_user(db_session)
        make_sandbox(db_session, user_a, status=SandboxStatus.RUNNING)
        sandbox_b = make_sandbox(db_session, user_b, status=SandboxStatus.RUNNING)
        make_sandbox(db_session, user_c, status=SandboxStatus.RUNNING)

        # user_b's push fatally fails; the other two succeed silently.
        stub = failing_sandbox_manager(
            fail_on={sandbox_b.id: FatalWriteError("Pod not found")}
        )
        monkeypatch.setattr("onyx.skills.push.get_sandbox_manager", lambda: stub)

        seeded_skill(
            slug=f"partial-{uuid4().hex[:6]}",
            public=True,
            bundle_files={"SKILL.md": "p\n"},
        )

        with caplog.at_level(logging.WARNING):
            # Must not raise even though one sandbox errors.
            push_skills_for_users({user_a.id, user_b.id, user_c.id}, db_session)

        # All three sandboxes were attempted (one failed, two succeeded).
        assert stub.write_files_to_sandbox_count == 3

        # A partial-failure warning was emitted (by push.py's per-failure line
        # or base.py's push_to_sandboxes aggregate line).
        warning_messages = [
            r.getMessage().lower()
            for r in caplog.records
            if r.levelno == logging.WARNING
        ]
        assert any("fail" in msg or "partial" in msg for msg in warning_messages), (
            f"Expected a partial-failure warning; got: {warning_messages!r}"
        )
