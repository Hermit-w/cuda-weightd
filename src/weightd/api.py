import os
from pathlib import Path

from weightd.client import WeightDClient
from weightd.core.bundle import WeightBundle


def _get_client() -> WeightDClient:
    """Get a WeightD client instance."""
    socket_path = os.environ.get(
        "WEIGHTD_SOCKET", str(Path.home() / ".cache" / "weightd" / "weightd.sock")
    )
    authkey = os.environ.get("WEIGHTD_AUTHKEY", "weightd-dev-only").encode()
    return WeightDClient(socket_path, authkey)


def list_shared_weights() -> list[str]:
    """List the available weight bundles on the WeightD server."""
    client = _get_client()
    return client.list()


def load_shared_weights(name: str) -> WeightBundle:
    """Load a weight bundle from the WeightD server."""
    client = _get_client()
    return client.get(name)
