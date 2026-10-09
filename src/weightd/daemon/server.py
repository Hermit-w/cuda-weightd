import logging
import os
from multiprocessing.connection import Connection, Listener
from pathlib import Path
from threading import Thread

from weightd.core.registry import BundleRegistry
from weightd.errors import BundleNotFoundError, InvalidRequestError, WeightDError
from weightd.ipc.protocol import (
    PROTOCOL_VERSION,
    Operation,
    make_response,
    validate_request,
)
from weightd.ipc.torch_transport import pack_bundle, unpack_bundle

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
        validate_request(request)
        op = request.get("op")
        match op:
            case Operation.PING:
                return make_response(ok=True, data="PONG")
            case Operation.LIST:
                return make_response(ok=True, data=self.registry.list())
            case Operation.GET:
                name = request.get("name")
                # TODO: do we need to check this parameter? The registry.get method will BundleNotFoundError, but
                # maybe InvalidRequestError is more appropriate for an empty or non-string name parameter.
                if not isinstance(name, str) or not name:
                    raise InvalidRequestError("Invalid or missing 'name' parameter")
                # register.get has handled the case where the bundle is not found and raises BundleNotFoundError
                bundle = self.registry.get(name)
                return make_response(ok=True, data=pack_bundle(bundle))
            case _:
                raise InvalidRequestError(f"Unknown operation: {op}")

    def handle_client(self, conn: Connection):
        """Handle a single connection."""
        with conn:
            try:
                request = conn.recv()
                try:
                    response = self.handle_request(request)
                except WeightDError as e:
                    logger.error(f"Error handling request: {e}")
                    response = make_response(ok=False, error=str(e))
                conn.send(response)
            except Exception as e:
                logger.exception("Error handling client connection: %s", e)
