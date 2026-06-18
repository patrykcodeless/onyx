"""Unit coverage for Kubernetes sandbox manager helpers."""

from __future__ import annotations

from uuid import UUID

import pytest

from onyx.server.features.build.sandbox.kubernetes.kubernetes_sandbox_manager import (
    KubernetesSandboxManager,
)

# K8s object names must be a DNS label: <=63 chars, lowercase alphanumerics
# and hyphens. ``_get_pod_name`` is the prefix for the pod, service, and the
# ``-opencode-auth`` secret, so the longest derived name must still fit.
_DNS_LABEL_MAX = 63
_LONGEST_SUFFIX = "-opencode-auth"


@pytest.mark.xfail(
    strict=True,
    reason="_get_pod_name truncates to the first 8 hex chars (32 bits), so "
    "distinct sandboxes can collide on a name prefix. Documented known bug.",
)
def test_pod_name_uses_full_uuid_not_first_8_chars() -> None:
    manager = KubernetesSandboxManager.__new__(KubernetesSandboxManager)

    # These two UUIDs share their first 8 hex chars; only the full UUID
    # distinguishes them.
    uuid_a = UUID("abc12345-0000-0000-0000-000000000001")
    uuid_b = UUID("abc12345-0000-0000-0000-000000000002")

    assert manager._get_pod_name(uuid_a) != manager._get_pod_name(uuid_b), (
        "pod name must encode the full UUID so distinct sandboxes do not "
        "collide on the first 8 hex chars"
    )


def test_pod_name_stays_within_dns_label_limit() -> None:
    manager = KubernetesSandboxManager.__new__(KubernetesSandboxManager)

    # All-f UUID is the worst case for length / character set.
    pod_name = manager._get_pod_name(UUID("ffffffff-ffff-ffff-ffff-ffffffffffff"))

    assert len(pod_name) + len(_LONGEST_SUFFIX) <= _DNS_LABEL_MAX
    assert pod_name == pod_name.lower()
    assert all(c.isalnum() or c == "-" for c in pod_name)
