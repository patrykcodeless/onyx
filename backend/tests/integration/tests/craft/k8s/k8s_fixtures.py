"""Shared Kubernetes fixtures and helpers for Craft integration tests."""

from __future__ import annotations

import json
import os
import shlex
import time
from collections.abc import Callable
from collections.abc import Generator
from collections.abc import Sequence
from contextlib import contextmanager
from contextlib import suppress
from dataclasses import dataclass
from dataclasses import field
from pathlib import PurePosixPath
from typing import TYPE_CHECKING
from uuid import UUID
from uuid import uuid4

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

if TYPE_CHECKING:
    from kubernetes import client as k8s_client_module

    from tests.integration.common_utils.test_models import DATestUser

from onyx.db.engine.sql_engine import get_session_with_current_tenant
from onyx.db.engine.sql_engine import SqlEngine
from onyx.db.enums import SandboxStatus
from onyx.db.models import ConnectorCredentialPair
from onyx.db.models import Credential
from onyx.db.models import Sandbox
from onyx.db.models import User
from onyx.server.features.build.configs import SANDBOX_NAMESPACE
from onyx.server.features.build.db.user_library import delete_user_file
from onyx.server.features.build.db.user_library import list_user_files
from onyx.server.features.build.sandbox.kubernetes.kubernetes_sandbox_manager import (
    KubernetesSandboxManager,
)
from onyx.utils.logger import setup_logger
from shared_configs.configs import POSTGRES_DEFAULT_SCHEMA_STANDARD_VALUE
from shared_configs.contextvars import CURRENT_TENANT_ID_CONTEXTVAR

logger = setup_logger()

_K8S_CRAFT_PATHS = (
    "backend/tests/integration/tests/craft/k8s/",
    "tests/integration/tests/craft/k8s/",
)


def _is_k8s_craft_request(request: pytest.FixtureRequest) -> bool:
    path = str(request.node.path).replace("\\", "/")
    return any(prefix in path for prefix in _K8S_CRAFT_PATHS)


def _sandbox_push_private_key() -> str:
    configured = os.environ.get("ONYX_SANDBOX_PUSH_PRIVATE_KEY")
    if configured:
        return configured

    private_key = Ed25519PrivateKey.generate()
    private_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )
    import base64

    return base64.b64encode(private_bytes).decode("ascii")


@pytest.fixture(scope="module", autouse=True)
def _sandbox_push_key(
    request: pytest.FixtureRequest,
) -> Generator[None, None, None]:
    # Module-scoped so it's set before ``_pool_pod`` (also module-scoped)
    # provisions its pod. ``sidecar_client`` imports the config value as a
    # module constant, so patch both the process env and the already-imported
    # modules.
    if not _is_k8s_craft_request(request):
        yield
        return

    from onyx.server.features.build import configs as build_configs
    from onyx.server.features.build.sandbox.kubernetes import sidecar_client

    push_key = _sandbox_push_private_key()
    mp = pytest.MonkeyPatch()
    mp.setenv("ONYX_SANDBOX_PUSH_PRIVATE_KEY", push_key)
    mp.setattr(build_configs, "SANDBOX_PUSH_PRIVATE_KEY", push_key)
    mp.setattr(sidecar_client, "SANDBOX_PUSH_PRIVATE_KEY", push_key)
    mp.setattr(sidecar_client, "_push_private_key", None)
    mp.setattr(sidecar_client, "_push_public_key_b64", None)
    try:
        yield
    finally:
        mp.undo()


# ---------------------------------------------------------------------------
# Pod-aware workspace proxy
#
# Migrated tests inspect files inside provisioned sandboxes via a ``Path``-like
# interface. With the local backend gone, those paths live inside a pod — but
# the call sites still want to write ``workspace.exists()``,
# ``(workspace / "managed" / "skills" / slug / "SKILL.md").read_bytes()``,
# etc. ``WorkspaceProxy`` mirrors the subset of ``pathlib.Path`` semantics the
# craft tests actually use; everything else raises so misuse fails loudly.
#
# All file operations go through ``pod_exec`` against the ``sandbox`` container,
# matching how production sandbox file ops work (read/list use exec; the
# managed/ tree is RO from the sandbox container but the tests read from it,
# which is fine).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkspaceProxy:
    """``Path``-shaped proxy for a sandbox pod's ``/workspace/<sandbox_id>``.

    Implements the subset of ``pathlib.Path`` used by Craft tests:
    ``/``, ``exists``, ``is_file``, ``is_symlink``, ``resolve``, ``read_bytes``,
    ``read_text``, ``rglob('*')``, ``name``. Everything else raises.

    Construct via :meth:`SandboxHandle.provision_api_user` — never directly.
    """

    _k8s_client: "k8s_client_module.CoreV1Api"
    _pod_name: str
    _sandbox_id: UUID
    _rel_parts: tuple[str, ...] = field(default_factory=tuple)

    # The "absolute path" inside the pod that this proxy represents. We use
    # ``/workspace`` (the per-pod root) + sandbox-id segment so the production
    # path layout (``managed/skills/...``, ``sessions/<id>/...``) matches what
    # the tests already write. Note: in the k8s manager, ``/workspace/managed``
    # and ``/workspace/sessions`` are pod-scoped, NOT sandbox-id-scoped, so we
    # drop the sandbox_id prefix unlike the old local layout.
    @property
    def _abs_posix(self) -> str:
        return (
            "/workspace/" + "/".join(self._rel_parts)
            if self._rel_parts
            else "/workspace"
        )

    @property
    def name(self) -> str:
        return self._rel_parts[-1] if self._rel_parts else "workspace"

    def __truediv__(self, segment: str | "WorkspaceProxy") -> "WorkspaceProxy":
        if isinstance(segment, WorkspaceProxy):
            raise TypeError("Cannot join two WorkspaceProxy instances")
        new_parts = self._rel_parts + tuple(
            p for p in PurePosixPath(segment).parts if p
        )
        return WorkspaceProxy(
            _k8s_client=self._k8s_client,
            _pod_name=self._pod_name,
            _sandbox_id=self._sandbox_id,
            _rel_parts=new_parts,
        )

    def _exec(self, command: str) -> str:
        from kubernetes.stream import stream as k8s_stream

        resp = k8s_stream(
            self._k8s_client.connect_get_namespaced_pod_exec,
            name=self._pod_name,
            namespace=SANDBOX_NAMESPACE,
            container="sandbox",
            command=["/bin/sh", "-c", command],
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False,
        )
        return str(resp) if resp is not None else ""

    def exists(self) -> bool:
        quoted = shlex.quote(self._abs_posix)
        # `test -e` returns true for files, dirs, and symlinks to anything.
        # We also accept dangling symlinks (`test -L`) so symlink presence
        # tests don't fall through to "missing" when the target is unset.
        out = self._exec(
            f"if [ -e {quoted} ] || [ -L {quoted} ]; then echo Y; else echo N; fi"
        )
        return "Y" in out

    def is_file(self) -> bool:
        out = self._exec(
            f"if [ -f {shlex.quote(self._abs_posix)} ]; then echo Y; else echo N; fi"
        )
        return "Y" in out

    def is_symlink(self) -> bool:
        out = self._exec(
            f"if [ -L {shlex.quote(self._abs_posix)} ]; then echo Y; else echo N; fi"
        )
        return "Y" in out

    def resolve(self) -> "WorkspaceProxy":
        """Best-effort symlink resolution via ``readlink -f``.

        Returned proxy points at the resolved absolute path. Tests use this
        only for symlink-target equality checks, so we return a proxy with
        the resolved path inlined as the ``_rel_parts`` tail.
        """
        out = self._exec(
            f"readlink -f {shlex.quote(self._abs_posix)} || echo {shlex.quote(self._abs_posix)}"
        )
        resolved = out.strip()
        # Strip the /workspace/ prefix if present; otherwise treat as absolute.
        # Split into individual segments either way so ``__truediv__`` and
        # ``_abs_posix`` produce correct results when callers continue to
        # navigate from the resolved proxy.
        if resolved.startswith("/workspace/"):
            rel = resolved[len("/workspace/") :]
        else:
            rel = resolved.lstrip("/")
        parts = tuple(p for p in rel.split("/") if p)
        return WorkspaceProxy(
            _k8s_client=self._k8s_client,
            _pod_name=self._pod_name,
            _sandbox_id=self._sandbox_id,
            _rel_parts=parts,
        )

    def read_bytes(self) -> bytes:
        import base64

        out = self._exec(
            f"base64 {shlex.quote(self._abs_posix)} 2>/dev/null || echo __MISSING__"
        )
        if "__MISSING__" in out:
            raise FileNotFoundError(self._abs_posix)
        return base64.b64decode(out.strip())

    def read_text(self) -> str:
        return self.read_bytes().decode("utf-8")

    def rglob(self, pattern: str) -> list["WorkspaceProxy"]:
        if pattern != "*":
            raise NotImplementedError(
                "WorkspaceProxy.rglob only supports '*' (used by craft tests)"
            )
        out = self._exec(
            f"find {shlex.quote(self._abs_posix)} -mindepth 1 2>/dev/null || true"
        )
        results: list[WorkspaceProxy] = []
        for line in out.splitlines():
            line = line.strip()
            if not line:
                continue
            # Each line is an absolute pod path; convert to ``_rel_parts``.
            if line.startswith("/workspace/"):
                rel = line[len("/workspace/") :]
            elif line == "/workspace":
                continue
            else:
                rel = line.lstrip("/")
            parts = tuple(p for p in rel.split("/") if p)
            results.append(
                WorkspaceProxy(
                    _k8s_client=self._k8s_client,
                    _pod_name=self._pod_name,
                    _sandbox_id=self._sandbox_id,
                    _rel_parts=parts,
                )
            )
        return results

    def __fspath__(self) -> str:
        return self._abs_posix

    def __str__(self) -> str:
        return self._abs_posix

    def __eq__(self, other: object) -> bool:
        if isinstance(other, WorkspaceProxy):
            return self._abs_posix == other._abs_posix
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self._abs_posix)


@dataclass(frozen=True)
class SandboxHandle:
    """Handle returned by the ``running_sandbox`` factory.

    Exposes the provisioned manager + IDs and resolves common workspace paths
    so call-sites stay short. Also supports ``provision_api_user(user)`` to add
    additional API-created users' sandboxes, mirroring the way API-driven push
    tests provision user cohorts.
    """

    manager: KubernetesSandboxManager
    sandbox_id: UUID
    session_id: UUID | None
    _k8s_client: "k8s_client_module.CoreV1Api"
    _register_extra: Callable[[UUID, "DATestUser"], None]
    _api_user: "DATestUser | None" = None

    @property
    def api_user(self) -> "DATestUser":
        if self._api_user is None:
            raise RuntimeError("SandboxHandle has no API user bound")
        return self._api_user

    @property
    def workspace_path(self) -> WorkspaceProxy:
        return WorkspaceProxy(
            _k8s_client=self._k8s_client,
            _pod_name=self.manager._get_pod_name(self.sandbox_id),
            _sandbox_id=self.sandbox_id,
        )

    def provision_api_user(
        self,
        api_user: "DATestUser",
    ) -> WorkspaceProxy:
        """Provision ``api_user`` through the deployed API."""
        sandbox_id, _session_id = _create_api_session_for_user(api_user)
        if sandbox_id == self.sandbox_id:
            if self._api_user is not None and api_user.id != self._api_user.id:
                raise AssertionError(
                    "API returned the pool sandbox for a different user; "
                    f"pool_user={self._api_user.id} api_user={api_user.id}"
                )
        else:
            self._register_extra(sandbox_id, api_user)

        return WorkspaceProxy(
            _k8s_client=self._k8s_client,
            _pod_name=self.manager._get_pod_name(sandbox_id),
            _sandbox_id=sandbox_id,
        )

    def provision_api_users(
        self,
        users: Sequence["DATestUser"],
    ) -> list[WorkspaceProxy]:
        """Provision API-created users through the deployed API in input order."""
        return [self.provision_api_user(user) for user in users]


def _create_api_user_and_session() -> tuple["DATestUser", UUID, UUID]:
    """Create a real API user/session and return (user, sandbox_id, session_id)."""
    from tests.integration.common_utils.managers.build_session import (
        BuildSessionManager,
    )
    from tests.integration.common_utils.managers.user import UserManager

    api_user = UserManager.create(name=f"craft-k8s-{uuid4().hex[:8]}")
    session = BuildSessionManager.create(api_user, headless=True)
    sandbox = session["sandbox"]
    assert sandbox is not None, f"Session response missing sandbox: {session!r}"
    assert sandbox["status"].upper() == SandboxStatus.RUNNING.value.upper()
    return api_user, UUID(sandbox["id"]), UUID(session["id"])


def _create_api_session_for_user(api_user: "DATestUser") -> tuple[UUID, UUID]:
    """Create a real API session for ``api_user`` and return (sandbox_id, session_id)."""
    from tests.integration.common_utils.managers.build_session import (
        BuildSessionManager,
    )

    session = BuildSessionManager.create(api_user, headless=True)
    sandbox = session["sandbox"]
    assert sandbox is not None, f"Session response missing sandbox: {session!r}"
    assert sandbox["status"].upper() == SandboxStatus.RUNNING.value.upper()
    return UUID(sandbox["id"]), UUID(session["id"])


def cleanup_api_user_sandbox_rows(user_id: UUID) -> None:
    try:
        with get_session_with_current_tenant() as session:
            for doc in list_user_files(session, user_id):
                delete_user_file(session, doc)
            for row in (
                session.query(ConnectorCredentialPair)
                .filter(ConnectorCredentialPair.creator_id == user_id)
                .all()
            ):
                session.delete(row)
            for row in session.query(Credential).filter(Credential.user_id == user_id):
                session.delete(row)
            for row in session.query(Sandbox).filter(Sandbox.user_id == user_id).all():
                session.delete(row)
            user_row = session.get(User, user_id)
            if user_row is not None:
                session.delete(user_row)
            session.commit()
    except Exception:
        logger.warning(
            "Failed to clean up Craft API test user %s", user_id, exc_info=True
        )


@contextmanager
def _provisioned_sandbox(
    manager: KubernetesSandboxManager,
    k8s_client: "k8s_client_module.CoreV1Api",
) -> Generator[tuple["DATestUser", UUID, UUID, str], None, None]:
    """The one way to get a real API-provisioned sandbox in a test.

    The proxy resolves pod identity via the DB (pod IP -> Sandbox.user_id), so
    a pod without a committed Sandbox row fails closed — gated egress 403s and
    snapshot uploads are silently dropped. Creating the session through the
    deployed API guarantees the app creates both the DB rows and the pod.
    Yields ``(api_user, sandbox_id, session_id, pod_name)``; tears down the pod
    and rows on exit.
    """
    api_user, sandbox_id, session_id = _create_api_user_and_session()
    user_id = UUID(api_user.id)
    try:
        pod_name = manager._get_pod_name(str(sandbox_id))
        try:
            yield api_user, sandbox_id, session_id, pod_name
        finally:
            try:
                manager.terminate(sandbox_id)
            except Exception:
                pass
            try:
                wait_for_pod_deletion(k8s_client, pod_name, SANDBOX_NAMESPACE)
            except Exception:
                pass
    finally:
        cleanup_api_user_sandbox_rows(user_id)


# ---------------------------------------------------------------------------
# Pool pod amortization
#
# Pod provisioning costs ~20s. With ~15 tests calling running_sandbox(), naive
# per-test provisioning would burn ~5 min of CI time on idle pod startup. The
# pool_pod fixture provisions exactly one pod per test module and lets each
# test reuse it via a fresh-session-id pattern + pre-test cleanup of the
# mutable workspace trees (managed/skills, managed/user_library, sessions/).
#
# Tests that need multiple distinct pods (cohort tests via
# ``SandboxHandle.provision_api_user``) still pay per-pod cost — those scenarios
# inherently require multiple pod identities, so amortization is impossible
# there. The savings come from amortizing the *primary* handle's pod.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _PoolPod:
    api_user: "DATestUser"
    sandbox_id: UUID
    pod_name: str
    manager: KubernetesSandboxManager
    k8s_client: "k8s_client_module.CoreV1Api"


def _cleanup_pool_workspace(
    k8s_client: "k8s_client_module.CoreV1Api",
    pod_name: str,
) -> None:
    """Wipe mutable trees on the pool pod before the next test runs.

    ``managed/`` is read-only in the sandbox app container but writable from
    the native init sidecar, so we exec via the sidecar for the skills +
    user_library subtrees. ``sessions/`` is on a shared emptyDir, writable from
    either container.
    """
    # managed/{skills,user_library} live under the RO mount — clean via sidecar.
    # ``find -mindepth 1 -delete`` removes only the directory's contents
    # (including dotfiles) without the ``.*`` glob expanding to ``.``/``..``.
    pod_exec(
        k8s_client,
        pod_name,
        SANDBOX_NAMESPACE,
        "find /workspace/managed/skills /workspace/managed/user_library "
        "-mindepth 1 -delete 2>/dev/null; true",
        container="sidecar",
    )
    # sessions/ is the per-session emptyDir tree.
    pod_exec(
        k8s_client,
        pod_name,
        SANDBOX_NAMESPACE,
        "find /workspace/sessions -mindepth 1 -delete 2>/dev/null; true",
        container="sandbox",
    )


@pytest.fixture(scope="module")
def _pool_pod(
    k8s_client: "k8s_client_module.CoreV1Api",
) -> Generator[_PoolPod, None, None]:
    """Module-scoped sandbox pod shared by all ``running_sandbox()`` calls.

    Yields a :class:`_PoolPod` whose ``sandbox_id`` is reused across every
    function-scoped ``running_sandbox()`` call in the module. The function
    fixture wipes mutable trees on entry, so each test sees a clean
    workspace.

    The pool sandbox's pod identity is fixed. For another user-owned pod, create
    a ``DATestUser`` through the API and call ``SandboxHandle.provision_api_user``.
    """
    from onyx.server.features.build.configs import SANDBOX_BACKEND
    from onyx.server.features.build.configs import SandboxBackend

    if SANDBOX_BACKEND != SandboxBackend.KUBERNETES:
        pytest.skip(
            "_pool_pod requires SANDBOX_BACKEND=kubernetes "
            "(run via pr-craft-k8s-tests.yml or against a local kind cluster)"
        )

    SqlEngine.init_engine(pool_size=10, max_overflow=5)
    token = CURRENT_TENANT_ID_CONTEXTVAR.set(POSTGRES_DEFAULT_SCHEMA_STANDARD_VALUE)
    manager = KubernetesSandboxManager()

    try:
        with _provisioned_sandbox(manager, k8s_client) as (
            api_user,
            pool_sandbox_id,
            _initial_session_id,
            pod_name,
        ):
            yield _PoolPod(
                api_user=api_user,
                sandbox_id=pool_sandbox_id,
                pod_name=pod_name,
                manager=manager,
                k8s_client=k8s_client,
            )
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)


@pytest.fixture(scope="function")
def running_sandbox(
    request: pytest.FixtureRequest,
) -> Callable[..., SandboxHandle]:
    """Factory: hand out a ``SandboxHandle`` bound to the module pool pod.

    Each call returns a handle backed by the shared :func:`_pool_pod`. The
    function fixture wipes ``/workspace/managed/skills``,
    ``/workspace/managed/user_library``, and ``/workspace/sessions`` on the
    pool pod before yielding, so every test sees a clean slate without
    paying the ~20s pod-provisioning cost. See module docstring above
    ``_PoolPod`` for the amortization rationale.

    Migration history: this fixture previously bound to
    ``LocalSandboxManager`` against ``tmp_path``. It now wraps a real
    ``KubernetesSandboxManager`` pool pod against the kind cluster. The fixture
    self-gates on ``SANDBOX_BACKEND == KUBERNETES`` and ``pytest.skip``s
    otherwise. Tests consuming it live in the K8s integration directory and run
    in the dedicated ``pr-craft-k8s-tests.yml`` lane.

    Extra user-owned pods are provisioned explicitly through
    ``SandboxHandle.provision_api_user``.
    """
    from onyx.server.features.build.configs import SANDBOX_BACKEND
    from onyx.server.features.build.configs import SandboxBackend

    if SANDBOX_BACKEND != SandboxBackend.KUBERNETES:
        pytest.skip(
            "running_sandbox fixture requires SANDBOX_BACKEND=kubernetes "
            "(run via pr-craft-k8s-tests.yml or against a local kind cluster)"
        )
    pool: _PoolPod = request.getfixturevalue("_pool_pod")

    # Pre-test cleanup of mutable trees on the pool pod.
    _cleanup_pool_workspace(pool.k8s_client, pool.pod_name)

    # Track per-test pods provisioned via SandboxHandle.provision_api_user so
    # teardown can terminate them. The pool pod itself is NOT terminated
    # here — that's the module fixture's job.
    extra_sandbox_user_ids: dict[UUID, UUID] = {}

    def _register_extra(sandbox_id: UUID, api_user: "DATestUser") -> None:
        extra_sandbox_user_ids[sandbox_id] = UUID(api_user.id)

    def _make(
        with_session: bool = False,
    ) -> SandboxHandle:
        session_id: UUID | None = None
        sandbox_id = pool.sandbox_id
        api_user: "DATestUser | None" = pool.api_user
        if with_session:
            sandbox_id, session_id = _create_api_session_for_user(pool.api_user)
            if sandbox_id != pool.sandbox_id:
                _register_extra(sandbox_id, pool.api_user)

        def _cleanup() -> None:
            for extra_id, user_id in extra_sandbox_user_ids.items():
                try:
                    pool.manager.terminate(extra_id)
                except Exception:
                    pass
                try:
                    wait_for_pod_deletion(
                        pool.k8s_client,
                        pool.manager._get_pod_name(extra_id),
                        SANDBOX_NAMESPACE,
                    )
                except Exception:
                    pass
                cleanup_api_user_sandbox_rows(user_id)
            # Pool pod is NOT terminated; the module fixture owns its
            # lifecycle. We deliberately leave mutable trees in place — the
            # next test's pre-yield cleanup wipes them.

        request.addfinalizer(_cleanup)

        return SandboxHandle(
            manager=pool.manager,
            sandbox_id=sandbox_id,
            session_id=session_id,
            _k8s_client=pool.k8s_client,
            _register_extra=_register_extra,
            _api_user=api_user,
        )

    return _make


# ---------------------------------------------------------------------------
# Kubernetes helpers (Part V.1)
#
# These are imported by the K8s integration test modules (test_kubernetes_sandbox.py,
# test_snapshot_restore.py, test_kubernetes_sandbox_file_ops.py). They never
# run against the deleted local backend; consumers gate execution behind a
# module-level ``pytestmark`` that skips when SANDBOX_BACKEND != KUBERNETES.
# The helpers are defined at module scope here so they can be imported as
# top-level callables.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def k8s_client() -> "k8s_client_module.CoreV1Api":
    """Session-scope CoreV1Api client.

    Only meaningful inside tests gated by
    ``pytest.mark.skipif(SANDBOX_BACKEND != KUBERNETES, ...)``. The fixture
    itself does not enforce that gate — module-level ``pytestmark`` does.
    """
    from kubernetes import client as k8s_client_module

    from onyx.server.features.build.sandbox.kubernetes.k8s_client import (
        load_kube_config,
    )

    load_kube_config()
    return k8s_client_module.CoreV1Api()


def pod_exec(
    client: "k8s_client_module.CoreV1Api",
    pod_name: str,
    namespace: str,
    command: str,
    container: str = "sandbox",
) -> str:
    """Run a one-shot command in a pod container; return combined output.

    Defaults to the ``sandbox`` container. Pass ``container="sidecar"`` for
    operations that need to write to ``/workspace/managed/`` (read-only in
    the sandbox container) or inspect the sidecar's environment.

    ``command`` is a shell-string run via ``/bin/sh -c``.
    """
    from kubernetes.stream import stream as k8s_stream

    argv = ["/bin/sh", "-c", command]
    resp = k8s_stream(
        client.connect_get_namespaced_pod_exec,
        name=pod_name,
        namespace=namespace,
        container=container,
        command=argv,
        stderr=True,
        stdin=False,
        stdout=True,
        tty=False,
    )
    return str(resp) if resp is not None else ""


def pod_exec_async(
    client: "k8s_client_module.CoreV1Api",
    pod_name: str,
    namespace: str,
    url: str,
    output_path: str,
    *,
    method: str = "POST",
    headers: dict[str, str] | None = None,
    body: str | None = None,
    body_file: str | None = None,
    max_time_s: int = 240,
    container: str = "sandbox",
    proxy_session_id: str | None = None,
) -> None:
    """Kick off a background sandbox-side ``curl``, writing ``{status}\\n{body}``
    to a tempfile only after curl exits. Returns immediately; poll via
    ``wait_for_pod_exec_output`` (a valid leading integer means done).

    ``proxy_session_id`` mimics the ``session-proxy-tag`` opencode plugin,
    tagging the request with the session id as ``Proxy-Authorization`` userinfo
    so the proxy can resolve the session. Omit it to exercise the untagged,
    fail-closed gate path.

    ``body_file`` (mutually exclusive with ``body``) sends the body from an
    in-pod path via ``--data-binary @path``; use this for payloads big enough to
    trip the apiserver's exec URL size limit (e.g. > 1 MiB).
    """
    if body is not None and body_file is not None:
        raise ValueError("pass either body or body_file, not both")
    header_args = ""
    for key, value in (headers or {}).items():
        header_args += f" -H {json.dumps(f'{key}: {value}')}"
    if body is not None:
        body_arg = f" --data {json.dumps(body)}"
    elif body_file is not None:
        body_arg = f" --data-binary @{body_file}"
    else:
        body_arg = ""
    # Override the ambient proxy with the session id as basic-auth userinfo.
    proxy_arg = (
        f" -x {json.dumps(f'http://{proxy_session_id}@sandbox-proxy:8080')}"
        if proxy_session_id is not None
        else ""
    )
    script = (
        f"nohup sh -c '"
        f"curl -s -X {method}{header_args}{body_arg}{proxy_arg} "
        f"--max-time {max_time_s} "
        f'-o {output_path}.body -w "%{{http_code}}" {json.dumps(url)} '
        f"> {output_path}.code 2>&1; "
        f'{{ cat {output_path}.code; printf "\\n"; cat {output_path}.body; }} '
        f"> {output_path}"
        f"' > /dev/null 2>&1 &"
    )
    pod_exec(client, pod_name, namespace, script, container=container)


def wait_for_pod_deletion(
    client: "k8s_client_module.CoreV1Api",
    pod_name: str,
    namespace: str = SANDBOX_NAMESPACE,
    max_attempts: int = 30,
) -> None:
    """Wait until the pod is fully gone (404) or in a terminating state."""
    from kubernetes.client.rest import ApiException

    for _ in range(max_attempts):
        try:
            pod = client.read_namespaced_pod(name=pod_name, namespace=namespace)
            if pod.metadata.deletion_timestamp is not None:
                time.sleep(1)
                continue
            time.sleep(1)
        except ApiException as e:
            if e.status == 404:
                return
            raise
    raise RuntimeError(
        f"Pod {pod_name} in namespace {namespace} was not deleted "
        f"after {max_attempts} attempts"
    )


def wait_for_pod_exec_output(
    client: "k8s_client_module.CoreV1Api",
    pod_name: str,
    output_path: str,
    timeout_s: float,
    namespace: str = SANDBOX_NAMESPACE,
    container: str = "sandbox",
) -> tuple[int, str]:
    """Poll the ``pod_exec_async`` tempfile until it appears, returning
    ``(status_code, body)`` parsed from its ``{status}\\n{body}`` layout.
    Raises on timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        raw = pod_exec(
            client,
            pod_name,
            namespace,
            f"cat {output_path} 2>/dev/null || true",
            container=container,
        )
        if raw:
            head, _, rest = raw.partition("\n")
            head = head.strip()
            if head.isdigit():
                return int(head), rest
        time.sleep(2)
    raise RuntimeError(
        f"pod_exec output {output_path} on pod {pod_name} did not arrive within "
        f"{timeout_s:.1f}s"
    )


def wait_for_proxy_redeploy(
    client: "k8s_client_module.CoreV1Api",
    timeout_s: float = 120,
) -> None:
    """Wait until the sandbox-proxy Deployment reports a ready replica.

    Used after a proxy-pod respawn so the next test doesn't hit a half-baked
    Deployment. Polls both Deployment status and pod-level readiness.
    """
    from kubernetes import client as k8s_client_module

    from onyx.server.features.build.configs import SANDBOX_PROXY_NAMESPACE

    proxy_component_label = "app.kubernetes.io/component=sandbox-proxy"
    apps_v1 = k8s_client_module.AppsV1Api()
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        deployments = apps_v1.list_namespaced_deployment(
            namespace=SANDBOX_PROXY_NAMESPACE,
            label_selector=proxy_component_label,
        )
        for deploy in deployments.items or []:
            ready = deploy.status.ready_replicas or 0
            desired = (
                deploy.spec.replicas if deploy.spec and deploy.spec.replicas else 1
            )
            if ready >= desired:
                pods = client.list_namespaced_pod(
                    namespace=SANDBOX_PROXY_NAMESPACE,
                    label_selector=proxy_component_label,
                )
                ready_pods = [
                    p
                    for p in (pods.items or [])
                    if any(cs.ready for cs in (p.status.container_statuses or []))
                ]
                if ready_pods:
                    return
        time.sleep(2)
    raise RuntimeError(
        f"sandbox-proxy Deployment did not return to ready within {timeout_s:.1f}s"
    )


# ---------------------------------------------------------------------------
# K8s shared fixtures (canonical home for k8s_manager + live_pod).
#
# Both fixtures previously lived (duplicated) in test_kubernetes_sandbox.py
# and test_snapshot_restore.py. Centralising them here lets each K8s test
# module just consume the fixture by name. Modules still set their own
# ``pytestmark = pytest.mark.skipif(SANDBOX_BACKEND != KUBERNETES, ...)`` —
# the fixtures themselves do not gate.
#
# The cluster check happens implicitly when ``k8s_client`` or the manager makes
# its first API call.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="function")
def k8s_manager() -> Generator[KubernetesSandboxManager, None, None]:
    """Initialise DB engine + tenant context and return the K8s manager.

    Consumer modules must gate themselves on
    ``SANDBOX_BACKEND == KUBERNETES`` via ``pytestmark``.
    """
    SqlEngine.init_engine(pool_size=10, max_overflow=5)
    token = CURRENT_TENANT_ID_CONTEXTVAR.set(POSTGRES_DEFAULT_SCHEMA_STANDARD_VALUE)
    try:
        yield KubernetesSandboxManager()
    finally:
        CURRENT_TENANT_ID_CONTEXTVAR.reset(token)


@pytest.fixture(scope="function")
def pool_session(
    _pool_pod: _PoolPod,
) -> tuple[UUID, UUID, str]:
    """Fresh API-created session on the module pool pod.

    Same return shape as ``live_pod`` (``sandbox_id, session_id,
    pod_name``) but reuses the module-scoped pool pod instead of
    provisioning + tearing down a fresh one per test. Saves ~14s of pod
    startup per test.

    Per call: wipes mutable trees on the pool pod
    (``/workspace/managed/skills``, ``/workspace/managed/user_library``,
    ``/workspace/sessions``) via :func:`_cleanup_pool_workspace`, then
    sets up a fresh session workspace with a new ``session_id``. The
    returned ``sandbox_id`` is the pool pod's stable ID.

    Use this for any test that needs a sandbox pod + a session and does
    NOT terminate / restart / re-provision the pod itself. Tests that
    assert on pod lifecycle (terminate cleanup, restart count, IRSA
    env, RO mount) must keep using ``live_pod`` so they don't break
    state for subsequent tests in the module.
    """
    _cleanup_pool_workspace(_pool_pod.k8s_client, _pool_pod.pod_name)
    sandbox_id, session_id = _create_api_session_for_user(_pool_pod.api_user)
    if sandbox_id != _pool_pod.sandbox_id:
        # The pool invariant broke (e.g. the pool pod was terminated externally
        # and the API provisioned a fresh one). Terminate the stray pod so it
        # doesn't leak, then fail loudly. Its DB rows are swept by the _pool_pod
        # teardown since it shares the pool's api_user.
        with suppress(Exception):
            _pool_pod.manager.terminate(sandbox_id)
        pytest.fail(
            f"pool_session: API returned a new sandbox {sandbox_id!r} instead "
            f"of the pool pod {_pool_pod.sandbox_id!r}; the pool pod may have "
            "been terminated externally."
        )
    return sandbox_id, session_id, _pool_pod.pod_name


@pytest.fixture(scope="function")
def live_pod(
    k8s_manager: KubernetesSandboxManager,
    k8s_client: "k8s_client_module.CoreV1Api",
) -> Generator[tuple[UUID, UUID, str], None, None]:
    """Provision a fresh sandbox + session pod, torn down on exit.

    Yields ``(sandbox_id, session_id, pod_name)``. Prefer ``pool_session``
    unless the test mutates pod-level state (lifecycle assertions, terminate).
    """
    with _provisioned_sandbox(k8s_manager, k8s_client) as (
        _api_user,
        sandbox_id,
        session_id,
        pod_name,
    ):
        yield sandbox_id, session_id, pod_name


@pytest.fixture(scope="function")
def provisioned_sandbox(
    k8s_manager: KubernetesSandboxManager,
    k8s_client: "k8s_client_module.CoreV1Api",
) -> Generator[tuple[UUID, str], None, None]:
    """A provisioned sandbox (committed rows + pod), without a session.

    For tests that need a real egressing pod as a precondition but not a
    session workspace (e.g. the egress proxy test). Backed by the same
    ``_provisioned_sandbox`` primitive, so the sandbox is identity-resolvable
    by the proxy. Yields ``(sandbox_id, pod_name)``.
    """
    with _provisioned_sandbox(k8s_manager, k8s_client) as (
        _api_user,
        sandbox_id,
        _session_id,
        pod_name,
    ):
        yield sandbox_id, pod_name
