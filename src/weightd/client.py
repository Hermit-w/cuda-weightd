from multiprocessing.connection import Client

from weightd.core.bundle import WeightBundle
from weightd.ipc.protocol import Operation, make_request
from weightd.ipc.torch_transport import recv_bundle


class WeightDClient:
    """WeightD client."""

    def __init__(self, socket_path: str, authkey: bytes):
        self.socket_path = socket_path
        self.authkey = authkey

    def _connect(self):
        """Connect to the WeightD server."""
        return Client(
            address=self.socket_path,
            family="AF_UNIX",
            authkey=self.authkey,
        )

    @staticmethod
    def _check_response(response: dict):
        """Check the response from the server."""
        if not isinstance(response, dict):
            raise ValueError("Invalid response format: expected a dictionary")

        if not response.get("ok", False):
            error_message = response.get("error", "Unknown error")
            raise RuntimeError(f"Server error: {error_message}")

    def ping(self) -> bool:
        """Ping the WeightD server."""
        with self._connect() as conn:
            conn.send(make_request(Operation.PING))
            response = conn.recv()
            self._check_response(response)
            return response.get("data") == "PONG"

    def list(self) -> list[str]:
        """List the available weight bundles on the server."""
        with self._connect() as conn:
            conn.send(make_request(Operation.LIST))
            response = conn.recv()
            self._check_response(response)
            return response.get("data", [])

    def get(self, name: str) -> WeightBundle:
        """Get a weight bundle from the server."""
        with self._connect() as conn:
            conn.send(make_request(Operation.GET, name=name))
            response = conn.recv()
            self._check_response(response)
            return recv_bundle(conn)
