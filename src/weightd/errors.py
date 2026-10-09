class WeightDError(Exception):
    """Base class for WeightD server errors."""

    pass


class BundleExistsError(WeightDError):
    """Raised when a weight bundle with the same name already exists."""

    pass


class BundleNotFoundError(WeightDError):
    """Raised when a requested weight bundle is not found."""

    pass


class InvalidRequestError(WeightDError):
    """Raised when a request to the server is invalid."""

    pass


class ProtocolVersionError(WeightDError):
    """Raised when there is a protocol error in communication with the server."""

    pass
