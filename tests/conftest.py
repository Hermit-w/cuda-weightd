from pathlib import Path

import pytest
import torch


@pytest.fixture(scope="session")
def cuda_device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    return torch.device("cuda:0")


@pytest.fixture
def socket_path(tmp_path: Path) -> str:
    return str(tmp_path / "weightd.sock")
