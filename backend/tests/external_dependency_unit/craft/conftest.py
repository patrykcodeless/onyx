"""External-dependency Craft fixtures.

See ``docs/craft/tests/coverage-and-overview.md`` for the contract these
fixtures honor and the broader test layer model. The k8s integration suite
intentionally uses its own fixture plugin so method-level DB factories and
stubs do not leak into full integration tests.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Iterable
from typing import Any
from uuid import UUID
from uuid import uuid4

import pytest
from fastapi_users.password import PasswordHelper
from sqlalchemy import text
from sqlalchemy.orm import Session

from onyx.configs.constants import FileOrigin
from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.engine.sql_engine import SqlEngine
from onyx.db.enums import AccountType
from onyx.db.enums import BuildSessionStatus
from onyx.db.enums import SandboxStatus
from onyx.db.llm import fetch_default_llm_model
from onyx.db.llm import fetch_existing_llm_provider
from onyx.db.llm import remove_llm_provider
from onyx.db.llm import update_default_provider
from onyx.db.llm import upsert_llm_provider
from onyx.db.models import BuildSession
from onyx.db.models import Sandbox
from onyx.db.models import Skill
from onyx.db.models import Skill__UserGroup
from onyx.db.models import User
from onyx.db.models import User__UserGroup
from onyx.db.models import UserGroup
from onyx.db.models import UserRole
from onyx.file_store.file_store import get_default_file_store
from onyx.llm.constants import LlmProviderNames
from onyx.server.features.build.db.sandbox import create_sandbox__no_commit
from onyx.server.features.build.db.sandbox import update_sandbox_status__no_commit
from onyx.server.features.build.session.manager import SessionManager
from onyx.server.manage.llm.models import LLMProviderUpsertRequest
from onyx.server.manage.llm.models import ModelConfigurationUpsertRequest
from shared_configs.configs import POSTGRES_DEFAULT_SCHEMA_STANDARD_VALUE
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR
from tests.common.craft.skill_table_isolation import restore_skill_tables
from tests.common.craft.skill_table_isolation import snapshot_skill_tables
from tests.common.craft.stubs import StubSandboxManager

# ---------------------------------------------------------------------------
# Skill-table isolation
# ---------------------------------------------------------------------------
#
# These tests run against the shared ``public`` schema (``POSTGRES_DEFAULT_SCHEMA_STANDARD_VALUE ==
# "public"``) — the very schema a self-hosted / local dev deployment uses. The
# fixtures and helpers below commit ``Skill`` / ``ExternalApp`` rows directly
# and nothing rolled them back, so every committed row leaked into the
# developer's live craft skill list (and into the next test's view of the
# table). Tests also delete/mutate the migration-seeded built-in rows
# (``pptx``, ``image-generation``, ``company-search``), corrupting them for the
# live app.
#
# The ``db_helpers`` contract states "the surrounding test owns transaction
# boundaries"; this autouse fixture is that boundary for the skill tables. It
# snapshots their committed state before each test and restores it afterward,
# so a run leaves these tables exactly as it found them (the canonical
# built-ins on a freshly-migrated DB).

_CRAFT_AUTOUSE_PATHS = (
    "backend/tests/external_dependency_unit/craft/",
    "tests/external_dependency_unit/craft/",
)
_CRAFT_EXT_DEP_PATHS = (
    "backend/tests/external_dependency_unit/craft/",
    "tests/external_dependency_unit/craft/",
)


def _request_path(request: pytest.FixtureRequest) -> str:
    return str(request.node.path).replace("\\", "/")


def _is_craft_request(request: pytest.FixtureRequest) -> bool:
    path = _request_path(request)
    return any(prefix in path for prefix in _CRAFT_AUTOUSE_PATHS)


def _is_ext_dep_craft_request(request: pytest.FixtureRequest) -> bool:
    path = _request_path(request)
    return any(prefix in path for prefix in _CRAFT_EXT_DEP_PATHS)


def _best_effort_delete(model: type[Any], ids: Iterable[Any]) -> None:
    """Delete rows by id on a fresh tenant session, swallowing errors.

    For fixture teardown: a failed delete (e.g. an FK that doesn't cascade)
    must not fail the test — at worst the row leaks, as it did before.
    """
    ids = [i for i in ids if i is not None]
    if not ids:
        return
    try:
        token = CURRENT_TENANT_ID_CONTEXTVAR.set(POSTGRES_DEFAULT_SCHEMA_STANDARD_VALUE)
        try:
            with get_session_with_current_tenant() as session:
                # Fail fast instead of hanging if another (uncommitted) test
                # session still holds locks on these rows.
                session.execute(text("SET lock_timeout = '10s'"))
                for row_id in ids:
                    row = session.get(model, row_id)
                    if row is not None:
                        session.delete(row)
                session.commit()
        finally:
            CURRENT_TENANT_ID_CONTEXTVAR.reset(token)
    except Exception:
        pass


def _best_effort_delete_memberships(group_ids: list[int]) -> None:
    """Delete User__UserGroup rows for the given groups (composite PK, so not
    deletable by ``_best_effort_delete``). Best-effort."""
    if not group_ids:
        return
    try:
        token = CURRENT_TENANT_ID_CONTEXTVAR.set(POSTGRES_DEFAULT_SCHEMA_STANDARD_VALUE)
        try:
            with get_session_with_current_tenant() as session:
                session.execute(text("SET lock_timeout = '10s'"))
                session.query(User__UserGroup).filter(
                    User__UserGroup.user_group_id.in_(group_ids)
                ).delete(synchronize_session=False)
                session.commit()
        finally:
            CURRENT_TENANT_ID_CONTEXTVAR.reset(token)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _isolate_skill_tables(
    request: pytest.FixtureRequest,
) -> Generator[None, None, None]:
    """Restore the skill tables to their pre-test state (see note above).

    Shares the test's ``db_session`` so there is a single transaction holder —
    no second connection that could block on row locks the test still holds.
    """
    if not _is_craft_request(request):
        yield
        return

    request.getfixturevalue("tenant_context")
    db_session = request.getfixturevalue("db_session")
    snapshot = snapshot_skill_tables(db_session)
    yield
    # Drop any uncommitted state a failing/early-exiting test left open before
    # reconciling against the committed baseline.
    db_session.rollback()
    restore_skill_tables(db_session, snapshot)


@pytest.fixture(scope="module", autouse=True)
def _seed_default_llm_provider(
    request: pytest.FixtureRequest,
) -> Generator[None, None, None]:
    """Seed a default LLM provider so the real provisioning path resolves one.

    No-op (and no teardown) if the DB already has a default. The fake key is
    never invoked — tests forward the resolved config to ``provision()`` only.
    """
    if not _is_ext_dep_craft_request(request):
        yield
        return

    SqlEngine.init_engine(pool_size=10, max_overflow=5)
    token = CURRENT_TENANT_ID_CONTEXTVAR.set(POSTGRES_DEFAULT_SCHEMA_STANDARD_VALUE)
    seeded_name: str | None = None
    try:
        with get_session_with_current_tenant() as session:
            if fetch_default_llm_model(session) is None:
                seeded_name = f"craft-ci-default-{uuid4().hex[:8]}"
                provider = upsert_llm_provider(
                    LLMProviderUpsertRequest(
                        name=seeded_name,
                        provider=LlmProviderNames.OPENAI,
                        api_key="sk-craft-ci-not-used",
                        api_key_changed=True,
                        model_configurations=[
                            ModelConfigurationUpsertRequest(
                                name="gpt-5-mini", is_visible=True
                            )
                        ],
                    ),
                    db_session=session,
                )
                update_default_provider(
                    provider_id=provider.id,
                    model_name="gpt-5-mini",
                    db_session=session,
                )
                session.commit()
        yield
    finally:
        if seeded_name is not None:
            with get_session_with_current_tenant() as session:
                existing = fetch_existing_llm_provider(
                    name=seeded_name, db_session=session
                )
                if existing is not None:
                    remove_llm_provider(session, existing.id)
                    session.commit()
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)


@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    """Create a database session for testing using the actual PostgreSQL database."""
    SqlEngine.init_engine(pool_size=10, max_overflow=5)
    with get_session_with_current_tenant() as session:
        yield session


@pytest.fixture(scope="function")
def tenant_context() -> Generator[None, None, None]:
    """Set up tenant context for testing."""
    token = CURRENT_TENANT_ID_CONTEXTVAR.set(POSTGRES_DEFAULT_SCHEMA_STANDARD_VALUE)
    try:
        yield
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)


@pytest.fixture(scope="function")
def test_user(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
) -> Generator[User, None, None]:
    """A committed test user; deleted on teardown (cascades its sandboxes,
    sessions, and memberships)."""
    password_helper = PasswordHelper()
    user = User(
        id=uuid4(),
        email=f"build_test_{uuid4().hex[:8]}@example.com",
        hashed_password=password_helper.hash(password_helper.generate()),
        is_active=True,
        is_superuser=False,
        is_verified=True,
        role=UserRole.EXT_PERM_USER,
        account_type=AccountType.EXT_PERM_USER,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    yield user
    # Release any uncommitted locks the test left on this session (e.g. an
    # ensure_sandbox_pat flush without commit) before the separate-session
    # delete below — otherwise its DELETE deadlocks on rows that cascade from
    # this user, since db_session (the lock holder) only tears down later (LIFO).
    db_session.rollback()
    _best_effort_delete(User, [user.id])


@pytest.fixture(scope="function")
def build_session(
    db_session: Session,
    test_user: User,
    tenant_context: None,  # noqa: ARG001
) -> BuildSession:
    """Create a test build session."""
    session = BuildSession(
        id=uuid4(),
        user_id=test_user.id,
        name="Test Build Session",
        status=BuildSessionStatus.ACTIVE,
    )
    db_session.add(session)
    db_session.commit()
    db_session.refresh(session)
    return session


@pytest.fixture(scope="function")
def sandbox(
    db_session: Session,
    test_user: User,
    tenant_context: None,  # noqa: ARG001
) -> Callable[..., Sandbox]:
    """Factory: create a ``Sandbox`` row for a user.

    Default owner is ``test_user``; default status is RUNNING. Pass ``user`` or
    ``status`` to override. Multiple calls (with distinct users) yield distinct
    rows.
    """

    def _make(
        user: User | None = None,
        status: SandboxStatus = SandboxStatus.RUNNING,
    ) -> Sandbox:
        owner = user or test_user
        # create_sandbox__no_commit starts at PROVISIONING; move to the asked status.
        row = create_sandbox__no_commit(db_session=db_session, user_id=owner.id)
        if status != SandboxStatus.PROVISIONING:
            update_sandbox_status__no_commit(db_session, row.id, status)
        db_session.commit()
        db_session.refresh(row)
        return row

    return _make


@pytest.fixture(scope="function")
def build_session_with_user(
    db_session: Session,
    test_user: User,
    sandbox: Callable[..., Sandbox],
    tenant_context: None,  # noqa: ARG001
) -> Callable[..., BuildSession]:
    """Factory: create a ``BuildSession`` tied to a user (and optional sandbox).

    Distinct from the existing ``build_session`` fixture (which is a single
    row, not a factory) because tests in Part V want to create multiple
    sessions per test.
    """

    def _make(
        user: User | None = None,
        status: BuildSessionStatus = BuildSessionStatus.ACTIVE,
        provision_sandbox: bool = False,
        name: str | None = None,
    ) -> BuildSession:
        owner = user or test_user
        if provision_sandbox:
            sandbox(user=owner)
        session_row = BuildSession(
            id=uuid4(),
            user_id=owner.id,
            name=name or "Test Build Session",
            status=status,
        )
        db_session.add(session_row)
        db_session.commit()
        db_session.refresh(session_row)
        return session_row

    return _make


@pytest.fixture(scope="function")
def granted_users(
    db_session: Session,
    request: pytest.FixtureRequest,
    tenant_context: None,  # noqa: ARG001
) -> Callable[..., dict[str, list[User]]]:
    """Factory: create users + sandboxes + groups in one call.

    Example
    -------
    ::

        cohort = granted_users(grants={"engineering": [None, None], "ops": [None]})

    Each value in the grants dict is interpreted as a list whose **length** is
    the number of users to create for that group. The factory creates the
    group if missing, creates fresh users for each slot, creates a sandbox per
    user (status=RUNNING), and links users to the group. Returns the
    realised mapping of group name → list of users.
    """
    password_helper = PasswordHelper()
    created_user_ids: list[UUID] = []
    created_group_ids: list[int] = []

    # Delete users (cascades their sandboxes / sessions / memberships), then
    # any membership left by a pre-existing user (its user_group_id FK is
    # RESTRICT, so it would block the group delete), then the groups we made.
    # Best-effort so teardown can't fail a test.
    def _cleanup() -> None:
        _best_effort_delete(User, created_user_ids)
        _best_effort_delete_memberships(created_group_ids)
        _best_effort_delete(UserGroup, created_group_ids)

    request.addfinalizer(_cleanup)

    def _make(grants: dict[str, list[User | None]]) -> dict[str, list[User]]:
        out: dict[str, list[User]] = {}
        for group_name, slots in grants.items():
            group = (
                db_session.query(UserGroup)
                .filter(UserGroup.name == group_name)
                .one_or_none()
            )
            if group is None:
                group = UserGroup(
                    name=group_name,
                    is_up_to_date=True,
                    is_up_for_deletion=False,
                    is_default=False,
                )
                db_session.add(group)
                db_session.commit()
                db_session.refresh(group)
                created_group_ids.append(group.id)

            created: list[User] = []
            for existing_user in slots:
                if existing_user is not None:
                    user = existing_user
                else:
                    password = password_helper.generate()
                    user = User(
                        id=uuid4(),
                        email=f"granted_{uuid4().hex[:8]}@example.com",
                        hashed_password=password_helper.hash(password),
                        is_active=True,
                        is_superuser=False,
                        is_verified=True,
                        role=UserRole.EXT_PERM_USER,
                        account_type=AccountType.EXT_PERM_USER,
                    )
                    db_session.add(user)
                    db_session.commit()
                    db_session.refresh(user)
                    created_user_ids.append(user.id)

                # One RUNNING sandbox per user.
                sandbox_row = create_sandbox__no_commit(
                    db_session=db_session, user_id=user.id
                )
                update_sandbox_status__no_commit(
                    db_session, sandbox_row.id, SandboxStatus.RUNNING
                )

                membership = User__UserGroup(
                    user_id=user.id,
                    user_group_id=group.id,
                    is_curator=False,
                )
                db_session.add(membership)

                created.append(user)

            db_session.commit()
            out[group_name] = created
        return out

    return _make


def _build_zip(files: dict[str, bytes | str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, content in files.items():
            data = content.encode("utf-8") if isinstance(content, str) else content
            zf.writestr(path, data)
    return buf.getvalue()


@pytest.fixture(scope="function")
def seeded_bundle() -> Callable[[dict[str, bytes | str]], bytes]:
    """Pure utility: pack a dict of paths → contents into a zip bundle.

    Returns the bytes; the caller decides where to put them.
    """
    return _build_zip


@pytest.fixture(scope="function")
def seeded_skill(
    db_session: Session,
    request: pytest.FixtureRequest,
    tenant_context: None,  # noqa: ARG001
) -> Callable[..., Skill]:
    """Factory: create a ``Skill`` row + its bundle in the file store.

    Convenience wrapper over the admin-skills create path. Tests that exercise
    the HTTP boundary should still go through the admin API; this factory is
    for tests that need a Skill row to be **present** without making HTTP
    calls. The Skill row is reclaimed by ``_isolate_skill_tables``; the bundle
    blob is deleted on teardown here.
    """
    file_store = get_default_file_store()
    file_store.initialize()
    bundle_file_ids: list[str] = []

    def _cleanup() -> None:
        for file_id in bundle_file_ids:
            try:
                file_store.delete_file(file_id, error_on_missing=False)
            except Exception:
                pass

    request.addfinalizer(_cleanup)

    def _make(
        slug: str,
        public: bool = False,
        groups: Iterable[UserGroup] | None = None,
        bundle_files: dict[str, bytes | str] | None = None,
        author_user_id: UUID | None = None,
    ) -> Skill:
        if bundle_files is None:
            bundle_files = {
                "SKILL.md": (
                    f"---\nname: {slug}\ndescription: Seeded skill {slug}\n---\n"
                ),
            }
        bundle_bytes = _build_zip(bundle_files)
        bundle_sha256 = hashlib.sha256(bundle_bytes).hexdigest()

        bundle_file_id = file_store.save_file(
            content=io.BytesIO(bundle_bytes),
            display_name=f"{slug}.zip",
            file_origin=FileOrigin.SKILL_BUNDLE,
            file_type="application/zip",
        )
        bundle_file_ids.append(bundle_file_id)

        skill = Skill(
            id=uuid4(),
            slug=slug,
            name=slug,
            description=f"Seeded skill {slug}",
            bundle_file_id=bundle_file_id,
            bundle_sha256=bundle_sha256,
            is_public=public,
            enabled=True,
            author_user_id=author_user_id,
        )
        db_session.add(skill)
        db_session.commit()
        db_session.refresh(skill)

        for group in groups or []:
            db_session.add(Skill__UserGroup(skill_id=skill.id, user_group_id=group.id))
        db_session.commit()
        return skill

    return _make


@pytest.fixture(scope="function")
def stub_sandbox_manager() -> StubSandboxManager:
    """Return a fresh ``StubSandboxManager`` per test."""
    return StubSandboxManager()


@pytest.fixture(scope="function")
def failing_sandbox_manager() -> Callable[..., StubSandboxManager]:
    """Factory variant: pre-configure a stub with a failure-injection map.

    Example
    -------
    ::

        stub = failing_sandbox_manager(
            fail_on={sandbox_id: FatalWriteError("nope")}
        )
    """

    def _make(
        fail_on: dict[UUID, Exception] | None = None,
    ) -> StubSandboxManager:
        stub = StubSandboxManager()
        if fail_on is not None:
            stub.write_files_to_sandbox_raises_for = dict(fail_on)
        return stub

    return _make


@pytest.fixture(scope="function")
def session_manager_with_stub(
    db_session: Session,
    tenant_context: None,  # noqa: ARG001
    stub_sandbox_manager: StubSandboxManager,
    monkeypatch: pytest.MonkeyPatch,
) -> SessionManager:
    """``SessionManager`` bound to the stub sandbox backend.

    Patches both ``session.manager.get_sandbox_manager`` (which
    ``SessionManager.__init__`` captures into ``self._sandbox_manager`` at
    construction time) AND ``sandbox.factory._sandbox_manager_instance`` so any
    deferred lookup also lands on the stub. The LLM lookup runs for real
    against the provider from ``_seed_default_llm_provider``.
    """
    monkeypatch.setattr(
        "onyx.server.features.build.session.manager.get_sandbox_manager",
        lambda: stub_sandbox_manager,
    )
    monkeypatch.setattr(
        "onyx.server.features.build.sandbox.factory._sandbox_manager_instance",
        stub_sandbox_manager,
    )
    sm = SessionManager(db_session)
    # Sanity: SessionManager captured the stub at construction.
    assert sm._sandbox_manager is stub_sandbox_manager
    return sm
