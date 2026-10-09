from enum import StrEnum

from weightd.errors import InvalidRequestError, ProtocolVersionError


class Operation(StrEnum):
    """IPC operation types."""

    PING = "PING"
    LIST = "LIST"
    GET = "GET"


PROTOCOL_VERSION = 1


def make_request(
    op: Operation,
    **kwargs,
) -> dict:
    return {
        "version": PROTOCOL_VERSION,
        "op": op.value,
        **kwargs,
    }


def validate_request(request: dict) -> None:
    if not isinstance(request, dict):
        raise InvalidRequestError("Request must be a dictionary.")

    version = request.get("version")
    if version != PROTOCOL_VERSION:
        raise ProtocolVersionError(
            f"Invalid protocol version: {version}. Expected: {PROTOCOL_VERSION}"
        )

    op = request.get("op")
    if op not in Operation.__members__:
        raise InvalidRequestError(f"Invalid operation: {op}.")


def make_response(
    *,
    ok: bool,
    data=None,
    error: str | None = None,
) -> dict:
    return {
        "ok": ok,
        "data": data,
        "error": error,
    }
