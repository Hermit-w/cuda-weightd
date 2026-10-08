from enum import StrEnum


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
