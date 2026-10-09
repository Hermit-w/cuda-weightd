from dataclasses import dataclass
from enum import StrEnum


class LeaseState(StrEnum):
    """Lease state enum."""

    ATTACHING = "attaching"
    ACTIVE = "active"
    RELEASED = "released"
    ABORTED = "aborted"
    CLIENT_LOST = "client_lost"


@dataclass(frozen=True)
class Lease:
    """Lease dataclass."""

    lease_id: str
    bundle_name: str
    client_pid: int
    state: LeaseState
    creation_time: float  # may be used for lease timeout handling in the future
    # may be used for lease cleanup handling in the future
    cleanup_confirmed: bool = False
