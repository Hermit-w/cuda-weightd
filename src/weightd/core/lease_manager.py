import threading
import time
import uuid
from dataclasses import replace

from weightd.core.lease import Lease, LeaseState
from weightd.errors import (
    InvalidLeaseStateError,
    LeaseNotFoundError,
    LeasePermissionError,
)


class LeaseManager:
    def __init__(self):
        self._leases: dict[str, Lease] = {}
        self._lock = threading.Lock()

    def acquire(self, bundle_name: str, client_pid: int) -> Lease:
        """Acquire a lease for a weight bundle."""
        with self._lock:
            lease_id = str(uuid.uuid4().hex)
            lease = Lease(
                lease_id=lease_id,
                bundle_name=bundle_name,
                client_pid=client_pid,
                state=LeaseState.ATTACHING,
                creation_time=time.time(),
            )
            self._leases[lease_id] = lease
            return lease

    def get(self, lease_id: str) -> Lease:
        """Get a lease by its ID."""
        with self._lock:
            lease = self._leases.get(lease_id)
            if lease is None:
                raise LeaseNotFoundError(f"Lease not found: {lease_id}")
            return lease

    def _get_unlocked(self, lease_id: str) -> Lease:
        """Get a lease by its ID without acquiring the lock. For internal use only."""
        lease = self._leases.get(lease_id)
        if lease is None:
            raise LeaseNotFoundError(f"Lease not found: {lease_id}")
        return lease

    def _transition_state_unlocked(
        self,
        lease_id: str,
        expected_state: LeaseState,
        target_state: LeaseState,
        *,
        cleanup_confirmed: bool = False,
    ) -> Lease:
        """Transition the state of a lease without acquiring the lock. For internal use only."""
        lease = self._get_unlocked(lease_id)
        if lease.state != expected_state:
            raise InvalidLeaseStateError(
                f"Invalid state transition: {lease.state} -> {target_state}"
                f"Expected state transition: {expected_state} -> {target_state}"
            )
        updated_lease = replace(
            lease, state=target_state, cleanup_confirmed=cleanup_confirmed
        )
        self._leases[lease_id] = updated_lease
        return updated_lease

    def commit(self, lease_id: str, client_pid: int) -> Lease:
        """Commit a lease, transitioning it from ATTACHING to ACTIVE."""
        with self._lock:
            self._verify_ownership_unlocked(lease_id, client_pid)
            return self._transition_state_unlocked(
                lease_id,
                expected_state=LeaseState.ATTACHING,
                target_state=LeaseState.ACTIVE,
            )

    def release(self, lease_id: str, client_pid: int) -> Lease:
        """Release a lease, transitioning it from ACTIVE to RELEASED."""
        with self._lock:
            self._verify_ownership_unlocked(lease_id, client_pid)
            lease = self._get_unlocked(lease_id)

            if lease.state == LeaseState.RELEASED:
                # If the lease is already released, we can return it as is.
                return lease
            return self._transition_state_unlocked(
                lease_id,
                expected_state=LeaseState.ACTIVE,
                target_state=LeaseState.RELEASED,
                cleanup_confirmed=True,
            )

    def abort(
        self, lease_id: str, client_pid: int, cleanup_confirmed: bool = False
    ) -> Lease:
        with self._lock:
            self._verify_ownership_unlocked(lease_id, client_pid)
            return self._transition_state_unlocked(
                lease_id,
                expected_state=LeaseState.ATTACHING,
                target_state=LeaseState.ABORTED,
                cleanup_confirmed=cleanup_confirmed,
            )

    def maybe_mark_client_lost(self, lease_id: str) -> Lease:
        """
        Once a client is detected to be lost (by pid check), we may need to mark it as CLIENT_LOST.
        However, a client may quit gracefully and release the lease, so we need to check the current state before marking it as CLIENT_LOST.
        If the lease is already in a terminal state (RELEASED, ABORTED, or CLIENT_LOST), we should not change its state.
        """

        with self._lock:
            lease = self._get_unlocked(lease_id)

            if lease.state in (
                LeaseState.RELEASED,
                LeaseState.ABORTED,
                LeaseState.CLIENT_LOST,
            ):
                # If the lease is already in a terminal state, we can return it as is.
                return lease

            # the client lost may be led by many reasons, we should not set cleanup_confirmed to True,
            # because we don't know if the client has cleaned up the resources or not.
            updated_lease = replace(
                lease, state=LeaseState.CLIENT_LOST, cleanup_confirmed=False
            )

            self._leases[lease_id] = updated_lease
            return updated_lease

    def maybe_mark_process_lost(self, client_pid: int) -> list[Lease]:
        """
        Mark all leases associated with a lost process as CLIENT_LOST.
        Once a process is detected to be lost (by pid check), we may need to mark it as CLIENT_LOST.
        However, a client may quit gracefully and release the lease, so we need to check the current state before marking it as CLIENT_LOST.
        If the lease is already in a terminal state (RELEASED, ABORTED, or CLIENT_LOST), we should not change its state.
        """

        with self._lock:
            leases_to_update = [
                lease
                for lease in self._leases.values()
                if lease.client_pid == client_pid
                and lease.state
                not in (
                    LeaseState.RELEASED,
                    LeaseState.ABORTED,
                    LeaseState.CLIENT_LOST,
                )
            ]

            updated_leases = []
            for lease in leases_to_update:
                updated_lease = replace(
                    lease, state=LeaseState.CLIENT_LOST, cleanup_confirmed=False
                )
                self._leases[lease.lease_id] = updated_lease
                updated_leases.append(updated_lease)

            return updated_leases

    def verify_ownership(self, lease_id: str, client_pid: int) -> None:
        """Verify that the given client_pid owns the lease."""
        with self._lock:
            lease = self._get_unlocked(lease_id)
            if lease.client_pid != client_pid:
                raise LeasePermissionError(
                    f"Client {client_pid} does not own lease {lease_id}"
                )

    def _verify_ownership_unlocked(self, lease_id: str, client_pid: int) -> None:
        """Verify that the given client_pid owns the lease without acquiring the lock. For internal use only."""
        lease = self._get_unlocked(lease_id)
        if lease.client_pid != client_pid:
            raise LeasePermissionError(
                f"Client {client_pid} does not own lease {lease_id}"
            )

    def list_leases(self, bundle_name: str | None = None) -> list[Lease]:
        with self._lock:
            return [
                lease
                for lease in self._leases.values()
                if lease.bundle_name == bundle_name or bundle_name is None
            ]

    def list_active_leases(self, bundle_name: str | None = None) -> list[Lease]:
        with self._lock:
            return [
                lease
                for lease in self._leases.values()
                if lease.state in (LeaseState.ACTIVE, LeaseState.ATTACHING)
                and (lease.bundle_name == bundle_name or bundle_name is None)
            ]

    def has_active_leases(self, bundle_name: str | None = None) -> bool:
        with self._lock:
            return any(
                lease.state in (LeaseState.ACTIVE, LeaseState.ATTACHING)
                for lease in self._leases.values()
                if lease.bundle_name == bundle_name or bundle_name is None
            )

    def can_unload_logically(self, bundle_name: str | None = None) -> bool:
        """
        Check if a weight bundle can be unloaded logically.
        A weight bundle can be unloaded logically if there are no active leases associated with it.
        """
        with self._lock:
            return all(
                lease.cleanup_confirmed
                for lease in self._leases.values()
                if lease.bundle_name == bundle_name or bundle_name is None
            )
