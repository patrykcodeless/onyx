"""Skill API push tests in the full Craft k8s integration lane.

Runs against real API/web/Celery/backing services, sandbox proxy, and sandbox
pods from the Helm-installed kind stack.
"""

from __future__ import annotations

import io
import time
import zipfile
from collections.abc import Callable
from collections.abc import Generator
from uuid import uuid4

import pytest

from onyx.server.features.build.configs import SANDBOX_BACKEND
from onyx.server.features.build.configs import SandboxBackend
from tests.integration.common_utils.managers.skill import SkillManager
from tests.integration.common_utils.managers.user import UserManager
from tests.integration.common_utils.managers.user_group import UserGroupManager
from tests.integration.common_utils.test_models import DATestSkill
from tests.integration.common_utils.test_models import DATestUser
from tests.integration.common_utils.test_models import DATestUserGroup
from tests.integration.tests.craft.k8s.k8s_fixtures import SandboxHandle
from tests.integration.tests.craft.k8s.k8s_fixtures import WorkspaceProxy

pytestmark = pytest.mark.skipif(
    SANDBOX_BACKEND != SandboxBackend.KUBERNETES,
    reason="K8s tests require SANDBOX_BACKEND=kubernetes; run in the dedicated K8s CI job.",
)


def _skill_file_path(
    workspace: WorkspaceProxy, slug: str, name: str = "SKILL.md"
) -> WorkspaceProxy:
    return workspace / "managed" / "skills" / slug / name


def _skills_dir(workspace: WorkspaceProxy) -> WorkspaceProxy:
    return workspace / "managed" / "skills"


def _bundle(slug: str, body: bytes | str, **extra_files: bytes | str) -> bytes:
    body_bytes = body.encode("utf-8") if isinstance(body, str) else body
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "SKILL.md",
            b"---\n"
            + f"name: {slug}\ndescription: {slug} integration test\n".encode("utf-8")
            + b"---\n"
            + body_bytes,
        )
        for path, content in extra_files.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            zf.writestr(path, data)
    return buf.getvalue()


def _create_skill(
    admin: DATestUser,
    slug: str,
    *,
    body: bytes | str,
    is_public: bool = False,
    group_ids: list[int] | None = None,
) -> DATestSkill:
    return SkillManager.create_custom(
        admin,
        slug=slug,
        is_public=is_public,
        group_ids=group_ids or [],
        bundle_bytes=_bundle(slug, body),
        filename=f"{slug}.zip",
    )


def _replace_bundle(
    admin: DATestUser,
    skill: DATestSkill,
    *,
    body: bytes | str,
) -> DATestSkill:
    return SkillManager.replace_bundle(
        skill,
        _bundle(skill.slug, body),
        admin,
    )


def _wait_for_bytes(
    path: WorkspaceProxy,
    expected: bytes,
    *,
    timeout_s: float = 20,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            if path.exists() and path.read_bytes().endswith(expected):
                return
        except Exception as e:
            last_error = e
        time.sleep(0.5)
    if last_error is not None:
        raise AssertionError(
            f"Timed out waiting for {path}: {last_error}"
        ) from last_error
    raise AssertionError(f"Timed out waiting for {path}")


def _wait_for_absent(path: WorkspaceProxy, *, timeout_s: float = 20) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not path.exists():
            return
        time.sleep(0.5)
    raise AssertionError(f"Timed out waiting for {path} to be absent")


def _create_users(count: int) -> list[DATestUser]:
    prefix = f"craft-k8s-skill-{uuid4().hex[:8]}"
    return [UserManager.create(name=f"{prefix}-{idx}") for idx in range(count)]


@pytest.fixture
def user_group_factory(
    k8s_admin_user: DATestUser,
) -> Generator[Callable[[str, list[str]], DATestUserGroup], None, None]:
    groups: list[DATestUserGroup] = []

    def _create(name: str, user_ids: list[str]) -> DATestUserGroup:
        group = UserGroupManager.create(
            k8s_admin_user,
            name=name,
            user_ids=user_ids,
        )
        groups.append(group)
        return group

    try:
        yield _create
    finally:
        for group in reversed(groups):
            UserGroupManager.delete(group, k8s_admin_user)


class TestSkillPush:
    def test_public_skill_lands_in_every_running_sandbox(
        self,
        k8s_admin_user: DATestUser,
        running_sandbox: Callable[..., SandboxHandle],
    ) -> None:
        handle = running_sandbox()
        users = _create_users(3)
        workspaces = handle.provision_api_users(users)

        slug = f"public-skill-{uuid4().hex[:6]}"
        skill = _create_skill(
            k8s_admin_user,
            slug,
            is_public=True,
            body="public skill body\n",
        )

        for workspace in workspaces:
            _wait_for_bytes(
                _skill_file_path(workspace, skill.slug),
                b"public skill body\n",
            )

    def test_private_skill_only_lands_in_granted_users_sandboxes(
        self,
        k8s_admin_user: DATestUser,
        running_sandbox: Callable[..., SandboxHandle],
        user_group_factory: Callable[[str, list[str]], DATestUserGroup],
    ) -> None:
        handle = running_sandbox()
        user_a, user_b, user_c = _create_users(3)
        [ws_a, ws_b, ws_c] = handle.provision_api_users([user_a, user_b, user_c])
        group = user_group_factory(
            f"engineering-{uuid4().hex[:6]}",
            [user_a.id],
        )

        slug = f"eng-only-{uuid4().hex[:6]}"
        skill = _create_skill(
            k8s_admin_user,
            slug,
            is_public=False,
            group_ids=[group.id],
            body="engineering only\n",
        )

        _wait_for_bytes(_skill_file_path(ws_a, skill.slug), b"engineering only\n")
        _wait_for_absent(_skill_file_path(ws_b, skill.slug))
        _wait_for_absent(_skill_file_path(ws_c, skill.slug))

    def test_disable_skill_removes_files_from_affected_sandboxes(
        self,
        k8s_admin_user: DATestUser,
        running_sandbox: Callable[..., SandboxHandle],
        user_group_factory: Callable[[str, list[str]], DATestUserGroup],
    ) -> None:
        handle = running_sandbox()
        [user] = _create_users(1)
        [workspace] = handle.provision_api_users([user])
        group = user_group_factory(
            f"disable-grp-{uuid4().hex[:6]}",
            [user.id],
        )

        slug = f"disable-me-{uuid4().hex[:6]}"
        skill = _create_skill(
            k8s_admin_user,
            slug,
            is_public=False,
            group_ids=[group.id],
            body="to be disabled\n",
        )
        _wait_for_bytes(_skill_file_path(workspace, skill.slug), b"to be disabled\n")

        SkillManager.patch_custom(skill, k8s_admin_user, enabled=False)

        _wait_for_absent(_skills_dir(workspace) / skill.slug)

    def test_grants_change_adds_to_newly_granted_and_removes_from_revoked(
        self,
        k8s_admin_user: DATestUser,
        running_sandbox: Callable[..., SandboxHandle],
        user_group_factory: Callable[[str, list[str]], DATestUserGroup],
    ) -> None:
        handle = running_sandbox()
        user_a, user_b = _create_users(2)
        [ws_a, ws_b] = handle.provision_api_users([user_a, user_b])
        group_x = user_group_factory(
            f"grp-x-{uuid4().hex[:6]}",
            [user_a.id],
        )
        group_y = user_group_factory(
            f"grp-y-{uuid4().hex[:6]}",
            [user_b.id],
        )

        slug = f"grants-flip-{uuid4().hex[:6]}"
        skill = _create_skill(
            k8s_admin_user,
            slug,
            is_public=False,
            group_ids=[group_x.id],
            body="shifting grants\n",
        )
        _wait_for_bytes(_skill_file_path(ws_a, skill.slug), b"shifting grants\n")
        _wait_for_absent(_skill_file_path(ws_b, skill.slug))

        SkillManager.replace_grants(skill, [group_y.id], k8s_admin_user)

        _wait_for_absent(_skill_file_path(ws_a, skill.slug))
        _wait_for_bytes(_skill_file_path(ws_b, skill.slug), b"shifting grants\n")

    def test_replace_bundle_propagates_new_content(
        self,
        k8s_admin_user: DATestUser,
        running_sandbox: Callable[..., SandboxHandle],
    ) -> None:
        handle = running_sandbox()
        [user] = _create_users(1)
        [workspace] = handle.provision_api_users([user])

        slug = f"versioned-{uuid4().hex[:6]}"
        skill = _create_skill(
            k8s_admin_user,
            slug,
            is_public=True,
            body="version one\n",
        )
        _wait_for_bytes(_skill_file_path(workspace, skill.slug), b"version one\n")

        _replace_bundle(k8s_admin_user, skill, body="version two\n")

        _wait_for_bytes(_skill_file_path(workspace, skill.slug), b"version two\n")

    def test_delete_skill_removes_directory_from_all_affected_sandboxes(
        self,
        k8s_admin_user: DATestUser,
        running_sandbox: Callable[..., SandboxHandle],
    ) -> None:
        handle = running_sandbox()
        user_a, user_b = _create_users(2)
        [ws_a, ws_b] = handle.provision_api_users([user_a, user_b])

        slug = f"to-delete-{uuid4().hex[:6]}"
        skill = _create_skill(
            k8s_admin_user,
            slug,
            is_public=True,
            body="will be deleted\n",
        )
        _wait_for_bytes(_skill_file_path(ws_a, skill.slug), b"will be deleted\n")
        _wait_for_bytes(_skill_file_path(ws_b, skill.slug), b"will be deleted\n")

        SkillManager.delete_custom(skill, k8s_admin_user)

        _wait_for_absent(_skills_dir(ws_a) / skill.slug)
        _wait_for_absent(_skills_dir(ws_b) / skill.slug)

    def test_user_with_overlapping_grants_receives_skill_once(
        self,
        k8s_admin_user: DATestUser,
        running_sandbox: Callable[..., SandboxHandle],
        user_group_factory: Callable[[str, list[str]], DATestUserGroup],
    ) -> None:
        handle = running_sandbox()
        [user] = _create_users(1)
        [workspace] = handle.provision_api_users([user])
        group_x = user_group_factory(
            f"dup-x-{uuid4().hex[:6]}",
            [user.id],
        )
        group_y = user_group_factory(
            f"dup-y-{uuid4().hex[:6]}",
            [user.id],
        )

        slug = f"dup-grants-{uuid4().hex[:6]}"
        skill = _create_skill(
            k8s_admin_user,
            slug,
            is_public=False,
            group_ids=[group_x.id, group_y.id],
            body="dedup\n",
        )

        _wait_for_bytes(_skill_file_path(workspace, skill.slug), b"dedup\n")
        skill_dir = _skills_dir(workspace) / skill.slug
        skill_files = [p for p in skill_dir.rglob("*") if p.is_file()]
        assert len(skill_files) == 1
        assert skill_files[0].name == "SKILL.md"
