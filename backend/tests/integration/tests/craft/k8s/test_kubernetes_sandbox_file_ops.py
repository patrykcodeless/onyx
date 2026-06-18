"""File-ops tests in the full Craft k8s integration lane.

Product file behavior is exercised through ``/build/sessions`` HTTP endpoints
against the Helm-installed API server. Direct manager calls below are limited
to Kubernetes-only lifecycle/snapshot contracts that have no product endpoint.
The lane runs with real API/web/Celery/backing services, sandbox proxy, and
sandbox pods.

Module is gated to the K8s CI lane via ``pytestmark`` (mirrors
``test_kubernetes_sandbox.py``).

Run with:

    SANDBOX_BACKEND=kubernetes python -m dotenv -f .vscode/.env run -- \\
        pytest backend/tests/integration/tests/craft/k8s/test_kubernetes_sandbox_file_ops.py -v
"""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import suppress
from uuid import UUID
from uuid import uuid4

import httpx
import pytest
from kubernetes import client

from onyx.db.enums import SandboxStatus
from onyx.server.features.build.configs import SANDBOX_BACKEND
from onyx.server.features.build.configs import SANDBOX_NAMESPACE
from onyx.server.features.build.configs import SandboxBackend
from onyx.server.features.build.sandbox.kubernetes.kubernetes_sandbox_manager import (
    KubernetesSandboxManager,
)
from tests.common.craft.payloads import default_llm_config
from tests.integration.common_utils.constants import API_SERVER_URL
from tests.integration.common_utils.http_client import client as http_client
from tests.integration.common_utils.managers.build_session import BuildSessionManager
from tests.integration.common_utils.test_models import DATestUser
from tests.integration.tests.craft.k8s.k8s_fixtures import pod_exec
from tests.integration.tests.craft.k8s.k8s_fixtures import SandboxHandle
from tests.integration.tests.craft.k8s.k8s_fixtures import wait_for_pod_deletion

pytestmark = pytest.mark.skipif(
    SANDBOX_BACKEND != SandboxBackend.KUBERNETES,
    reason="K8s tests require SANDBOX_BACKEND=kubernetes; run in the dedicated K8s CI job.",
)


def _artifact_response(user: DATestUser, session_id: UUID, path: str) -> httpx.Response:
    return http_client.get(
        f"{API_SERVER_URL}/build/sessions/{session_id}/artifacts/{path}",
        headers=user.headers,
        cookies=user.cookies,
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthCheck:
    def test_health_check_returns_true_for_provisioned_sandbox(
        self,
        k8s_manager: KubernetesSandboxManager,
        pool_session: tuple[UUID, UUID, str],
    ) -> None:
        sandbox_id, _, _ = pool_session
        assert k8s_manager.health_check(sandbox_id, timeout=10.0) is True

    def test_health_check_returns_false_after_terminate(
        self,
        k8s_manager: KubernetesSandboxManager,
        k8s_client: client.CoreV1Api,
    ) -> None:
        sandbox_id = uuid4()
        pod_name = k8s_manager._get_pod_name(sandbox_id)
        try:
            info = k8s_manager.provision(
                sandbox_id=sandbox_id,
                user_id=UUID("ee0dd46a-23dc-4128-abab-6712b3f4464c"),
                tenant_id="tenant_test",
                llm_config=default_llm_config(),
                onyx_pat="ci-test-pat",
            )
            assert info.status == SandboxStatus.RUNNING

            # Block until pod is healthy before terminating, otherwise the test
            # races the pod-ready path.
            for _ in range(15):
                if k8s_manager.health_check(sandbox_id, timeout=5.0):
                    break
                time.sleep(2)

            k8s_manager.terminate(sandbox_id)
            wait_for_pod_deletion(k8s_client, pod_name, SANDBOX_NAMESPACE)

            assert k8s_manager.health_check(sandbox_id, timeout=5.0) is False
        finally:
            with suppress(Exception):
                k8s_manager.terminate(sandbox_id)
            with suppress(Exception):
                wait_for_pod_deletion(k8s_client, pod_name, SANDBOX_NAMESPACE)


# ---------------------------------------------------------------------------
# List directory
# ---------------------------------------------------------------------------


class TestListDirectory:
    def test_list_directory_api_returns_entries(
        self,
        k8s_client: client.CoreV1Api,
        running_sandbox: Callable[..., SandboxHandle],
    ) -> None:
        handle = running_sandbox(with_session=True)
        assert handle.session_id is not None
        # ``list_directory`` walks ``sessions/{id}/{path}`` with ``ls -laL``.
        # The session root contains ``user_library`` (a symlink into the RO
        # managed/ mount) which fails ``-L`` dereference and trips the
        # function's ``ERROR_NOT_FOUND`` branch, so the contract in
        # production is "list a subpath under outputs/" (the docstring's
        # documented path). Seed + assert inside outputs/.
        outputs_dir = f"/workspace/sessions/{handle.session_id}/outputs"
        pod_exec(
            k8s_client,
            handle.manager._get_pod_name(handle.sandbox_id),
            SANDBOX_NAMESPACE,
            f"mkdir -p {outputs_dir}/subdir && echo content > {outputs_dir}/file.txt",
        )

        result = BuildSessionManager.list_files(
            handle.api_user,
            handle.session_id,
            "outputs",
        )

        entry_names = {entry["name"] for entry in result["entries"]}
        assert "file.txt" in entry_names
        assert "subdir" in entry_names


# ---------------------------------------------------------------------------
# Read file
# ---------------------------------------------------------------------------


class TestReadFile:
    def test_download_artifact_api_returns_contents(
        self,
        k8s_client: client.CoreV1Api,
        running_sandbox: Callable[..., SandboxHandle],
    ) -> None:
        handle = running_sandbox(with_session=True)
        assert handle.session_id is not None
        outputs_dir = f"/workspace/sessions/{handle.session_id}/outputs"
        pod_exec(
            k8s_client,
            handle.manager._get_pod_name(handle.sandbox_id),
            SANDBOX_NAMESPACE,
            f"mkdir -p {outputs_dir} && printf 'Hello, World!' > {outputs_dir}/test.txt",
        )

        result = BuildSessionManager.download_artifact(
            handle.api_user,
            handle.session_id,
            "outputs/test.txt",
        )
        assert result == b"Hello, World!"


# ---------------------------------------------------------------------------
# Delete file
# ---------------------------------------------------------------------------


class TestDeleteFile:
    def test_delete_file_api_removes_file(
        self,
        running_sandbox: Callable[..., SandboxHandle],
    ) -> None:
        handle = running_sandbox(with_session=True)
        assert handle.session_id is not None
        upload = BuildSessionManager.upload_file(
            handle.api_user,
            handle.session_id,
            "test.txt",
            b"content",
        )
        path = str(upload["path"])

        BuildSessionManager.delete_file(handle.api_user, handle.session_id, path)

        response = _artifact_response(handle.api_user, handle.session_id, path)
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Snapshot
# ---------------------------------------------------------------------------


class TestCreateSnapshot:
    """The full snapshot/restore round-trip lives in test_snapshot_restore.py.
    Here we pin the no-outputs short-circuit which the local manager could
    not exercise."""

    def test_create_snapshot_returns_none_when_session_has_no_outputs(
        self,
        k8s_manager: KubernetesSandboxManager,
        k8s_client: client.CoreV1Api,
        pool_session: tuple[UUID, UUID, str],
    ) -> None:
        sandbox_id, session_id, pod_name = pool_session

        # ``setup_session_workspace`` populates outputs/web/ with the Next.js
        # scaffold, so a freshly-set-up session is never literally empty.
        # Wipe the snapshot-eligible trees first to exercise the sidecar's
        # ``status="empty"`` short-circuit. (managed/* lives outside the
        # snapshot, so we don't need to clean it.)
        session_root = f"/workspace/sessions/{session_id}"
        pod_exec(
            k8s_client,
            pod_name,
            SANDBOX_NAMESPACE,
            f"rm -rf {session_root}/outputs {session_root}/attachments "
            "2>/dev/null; true",
        )

        result = k8s_manager.create_snapshot(
            sandbox_id, session_id, tenant_id="tenant_test"
        )

        assert result is None


# ---------------------------------------------------------------------------
# Terminate
# ---------------------------------------------------------------------------


class TestTerminate:
    def test_terminate_cleans_up_resources(
        self,
        k8s_manager: KubernetesSandboxManager,
        k8s_client: client.CoreV1Api,
    ) -> None:
        """``terminate`` deletes the pod (no exception, no left-over pod)."""
        sandbox_id = uuid4()
        pod_name = k8s_manager._get_pod_name(sandbox_id)
        try:
            k8s_manager.provision(
                sandbox_id=sandbox_id,
                user_id=UUID("ee0dd46a-23dc-4128-abab-6712b3f4464c"),
                tenant_id="tenant_test",
                llm_config=default_llm_config(),
                onyx_pat="ci-test-pat",
            )
            for _ in range(15):
                if k8s_manager.health_check(sandbox_id, timeout=5.0):
                    break
                time.sleep(2)

            k8s_manager.terminate(sandbox_id)
            wait_for_pod_deletion(k8s_client, pod_name, SANDBOX_NAMESPACE)
        finally:
            with suppress(Exception):
                k8s_manager.terminate(sandbox_id)
            with suppress(Exception):
                wait_for_pod_deletion(k8s_client, pod_name, SANDBOX_NAMESPACE)


# ---------------------------------------------------------------------------
# Upload file
# ---------------------------------------------------------------------------


class TestUploadFile:
    def test_upload_file_api_creates_file(
        self,
        running_sandbox: Callable[..., SandboxHandle],
    ) -> None:
        handle = running_sandbox(with_session=True)
        assert handle.session_id is not None
        content = b"Hello, World!"

        result = BuildSessionManager.upload_file(
            handle.api_user,
            handle.session_id,
            "test.txt",
            content,
        )

        assert result["path"] == "attachments/test.txt"

        readback = BuildSessionManager.download_artifact(
            handle.api_user,
            handle.session_id,
            "attachments/test.txt",
        )
        assert readback == content

    def test_upload_file_api_handles_collision(
        self,
        running_sandbox: Callable[..., SandboxHandle],
    ) -> None:
        handle = running_sandbox(with_session=True)
        assert handle.session_id is not None

        BuildSessionManager.upload_file(
            handle.api_user,
            handle.session_id,
            "collide.txt",
            b"first",
        )
        result = BuildSessionManager.upload_file(
            handle.api_user,
            handle.session_id,
            "collide.txt",
            b"second",
        )

        assert result["path"] == "attachments/collide_1.txt"
        assert (
            BuildSessionManager.download_artifact(
                handle.api_user,
                handle.session_id,
                "attachments/collide.txt",
            )
            == b"first"
        )
        assert (
            BuildSessionManager.download_artifact(
                handle.api_user,
                handle.session_id,
                "attachments/collide_1.txt",
            )
            == b"second"
        )

    def test_upload_first_file_injects_agents_md_attachments_section(
        self,
        k8s_client: client.CoreV1Api,
        running_sandbox: Callable[..., SandboxHandle],
    ) -> None:
        """First upload injects the attachments section into AGENTS.md;
        subsequent uploads don't duplicate it. Pins
        ``_ensure_agents_md_attachments_section``.
        """
        handle = running_sandbox(with_session=True)
        assert handle.session_id is not None
        pod_name = handle.manager._get_pod_name(handle.sandbox_id)
        agents_md_path = f"/workspace/sessions/{handle.session_id}/AGENTS.md"
        section_marker = "## Attachments (PRIORITY)"

        before = pod_exec(
            k8s_client, pod_name, SANDBOX_NAMESPACE, f"cat {agents_md_path}"
        )
        assert section_marker not in before, (
            "precondition: AGENTS.md should not yet contain the attachments section"
        )

        BuildSessionManager.upload_file(
            handle.api_user,
            handle.session_id,
            "first.txt",
            b"hello",
        )
        after_first = pod_exec(
            k8s_client, pod_name, SANDBOX_NAMESPACE, f"cat {agents_md_path}"
        )
        assert section_marker in after_first, (
            "first upload must inject the attachments section into AGENTS.md"
        )

        BuildSessionManager.upload_file(
            handle.api_user,
            handle.session_id,
            "second.txt",
            b"world",
        )
        after_second = pod_exec(
            k8s_client, pod_name, SANDBOX_NAMESPACE, f"cat {agents_md_path}"
        )
        assert after_second.count(section_marker) == 1, (
            "second upload should not duplicate the attachments section; "
            f"got {after_second.count(section_marker)} occurrences"
        )
