import threading

from weightd.core.lease_manager import LeaseManager
from weightd.core.registry import BundleRegistry
from weightd.errors import ServiceDrainingError


class WeightDService:
    """WeightD service."""

    def __init__(self, registry: BundleRegistry):
        self.lease_manager = LeaseManager()
        self.registry = registry
        self._draining = False
        self._lock = threading.Lock()

    def drain(self) -> bool:
        """Drain the service, preventing new leases from being acquired."""
        with self._lock:
            self._draining = True
            return self.lease_manager.can_unload_logically()

    def attach(self, bundle_name: str, client_pid: int):
        """Acquire a lease for a weight bundle."""
        with self._lock:
            if self._draining:
                raise ServiceDrainingError(
                    "Service is draining, cannot acquire new leases."
                )
            bundle = self.registry.get(bundle_name)
            lease = self.lease_manager.acquire(bundle_name, client_pid)
            return lease, bundle

    def commit(self, lease_id: str, client_pid: int):
        """Commit a lease, transitioning it from ATTACHING to ACTIVE."""
        self.lease_manager.commit(lease_id, client_pid)

    def release(self, lease_id: str, client_pid: int):
        """Release a lease, transitioning it from ACTIVE to RELEASED."""
        self.lease_manager.release(lease_id, client_pid)

    def abort(self, lease_id: str, client_pid: int):
        """Abort a lease, transitioning it to ABORTED."""
        self.lease_manager.abort(lease_id, client_pid)

    def client_lost(self, client_pid: int):
        """Mark all leases held by a client as CLIENT_LOST."""
        self.lease_manager.maybe_mark_process_lost(client_pid)
