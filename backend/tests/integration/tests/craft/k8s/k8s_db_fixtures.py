"""K8s-only Craft database fixtures.

These exist for setup/cleanup and for assertions that currently have no API
surface. Stub managers and method-level row factories live in
``tests.external_dependency_unit.craft.conftest`` for external-dependency unit tests.
"""

from __future__ import annotations

from collections.abc import Generator

import pytest
from sqlalchemy.orm import Session

from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.engine.sql_engine import SqlEngine
from shared_configs.configs import POSTGRES_DEFAULT_SCHEMA_STANDARD_VALUE
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR
from tests.common.craft.skill_table_isolation import restore_skill_tables
from tests.common.craft.skill_table_isolation import snapshot_skill_tables

_K8S_SKILL_ISOLATION_PATHS = (
    "backend/tests/integration/tests/craft/k8s/test_approval_gate.py",
    "backend/tests/integration/tests/craft/k8s/test_skill_push.py",
    "tests/integration/tests/craft/k8s/test_approval_gate.py",
    "tests/integration/tests/craft/k8s/test_skill_push.py",
)


def _needs_skill_table_isolation(request: pytest.FixtureRequest) -> bool:
    path = str(request.node.path).replace("\\", "/")
    return any(path.endswith(suffix) for suffix in _K8S_SKILL_ISOLATION_PATHS)


@pytest.fixture(autouse=True)
def _isolate_skill_tables(
    request: pytest.FixtureRequest,
) -> Generator[None, None, None]:
    if not _needs_skill_table_isolation(request):
        yield
        return

    request.getfixturevalue("tenant_context")
    db_session = request.getfixturevalue("db_session")
    snapshot = snapshot_skill_tables(db_session)
    yield
    db_session.rollback()
    restore_skill_tables(db_session, snapshot)


@pytest.fixture(scope="function")
def db_session() -> Generator[Session, None, None]:
    SqlEngine.init_engine(pool_size=10, max_overflow=5)
    with get_session_with_current_tenant() as session:
        yield session


@pytest.fixture(scope="function")
def tenant_context() -> Generator[None, None, None]:
    token = CURRENT_TENANT_ID_CONTEXTVAR.set(POSTGRES_DEFAULT_SCHEMA_STANDARD_VALUE)
    try:
        yield
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)
