import logging
import os
from multiprocessing.connection import Connection, Listener
from pathlib import Path
from threading import Thread

from weightd.core.registry import BundleRegistry
from weightd.errors import BundleNotFoundError, InvalidRequestError
from weightd.ipc.protocol import PROTOCOL_VERSION, Operation, make_response
from weightd.ipc.torch_transport import send_bundle

logger = logging.getLogger(__name__)


class WeightDServer:
    """WeightD server."""

    def __init__(self, registry: BundleRegistry, socket_path: str, authkey: bytes):
        self.registry = registry
        self.socket_path = socket_path
        self.authkey = authkey

    def run(self):
        """Run the server."""
        socket = Path(self.socket_path)
        if not socket.parent.exists():
            socket.parent.mkdir(parents=True, exist_ok=True)

        if socket.exists():
            raise FileExistsError(f"Socket already exists: {socket}")

        with Listener(
            address=self.socket_path,
            family="AF_UNIX",
            authkey=self.authkey,
        ) as listener:
            os.chmod(self.socket_path, 0o600)
            logger.info("WeightD Server listening on %s", self.socket_path)

            # loop to accept connections
            while True:
                conn = listener.accept()
                logger.info("Accepted connection.")
                # Handle the client in a separate thread
                thread = Thread(target=self.handle_client, args=(conn,), daemon=True)
                thread.start()

    def handle_request(self, request: dict) -> dict:
        """Handle a single request."""
        if not isinstance(request, dict):
            raise InvalidRequestError("Invalid request format: expected a dictionary")
        op = request.get("op")
        match op:
            case Operation.PING:
                return make_response(ok=True, data="PONG")
            case Operation.LIST:
                return make_response(ok=True, data=self.registry.list())
            case Operation.GET:
                name = request.get("name")
                if not isinstance(name, str) or not name:
                    raise InvalidRequestError("Invalid or missing 'name' parameter")
                # register.get has handled the case where the bundle is not found and raises BundleNotFoundError
                bundle = self.registry.get(name)
                return make_response(ok=True, data=bundle)
            case _:
                raise InvalidRequestError(f"Unknown operation: {op}")

    def handle_client(self, conn: Connection):
        """Handle a single connection."""
        with conn:
            try:
                request = conn.recv()

                if not isinstance(request, dict):
                    raise ValueError("Invalid request format: expected a dictionary")

                if request.get("version") != PROTOCOL_VERSION:
                    raise ValueError(
                        f"Protocol version mismatch: expected {PROTOCOL_VERSION}, got {request.get('version')}"
                    )

                op = request.get("op")
                logger.debug("Received request: %s", request)

                match op:
                    case Operation.PING:
                        conn.send(make_response(ok=True, data="PONG"))
                    case Operation.LIST:
                        conn.send(make_response(ok=True, data=self.registry.list()))
                    case Operation.GET:
                        name = request.get("name")
                        if not isinstance(name, str) or not name:
                            conn.send(
                                make_response(
                                    ok=False,
                                    error="Invalid or missing 'name' parameter",
                                )
                            )
                            return
                        bundle = self.registry.get(name)
                        if bundle is None:
                            conn.send(
                                make_response(
                                    ok=False,
                                    error=f"Bundle not found: {name}",
                                )
                            )
                        else:
                            send_bundle(conn, bundle)
                    case _:
                        conn.send(
                            make_response(
                                ok=False,
                                error=f"Unknown operation: {op}",
                            )
                        )
            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"Error handling request: {e}")
                conn.send(make_response(ok=False, error=str(e)))
            except (EOFError, OSError, ConnectionError):
                pass  # Client disconnected
