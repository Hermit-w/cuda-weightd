import logging
import os
from pathlib import Path

import torch

from weightd.core.bundle import WeightBundle
from weightd.core.registry import BundleRegistry
from weightd.daemon.server import WeightDServer
from weightd.logging import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. WeightD requires a CUDA-enabled GPU."
        )

    device = torch.device("cuda:0")

    bundle = WeightBundle(
        name="example_model",
        tensors={
            "layer.weight": torch.arange(
                16, dtype=torch.float32, device=device
            ).reshape(4, 4),
            "layer.bias": torch.ones(4, dtype=torch.float32, device=device),
        },
    )

    registry = BundleRegistry()
    registry.register(bundle)

    socket_path = os.environ.get(
        "WEIGHTD_SOCKET", str(Path.home() / ".cache" / "weightd" / "weightd.sock")
    )
    authkey = os.environ.get("WEIGHTD_AUTHKEY", "weightd-dev-only").encode()

    server = WeightDServer(registry, socket_path, authkey)

    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Shutting down WeightD server...")
    except Exception as e:
        logger.exception("WeightD server failed")
        raise SystemExit(1) from e
    finally:
        if os.path.exists(socket_path):
            os.unlink(socket_path)


if __name__ == "__main__":
    main()
