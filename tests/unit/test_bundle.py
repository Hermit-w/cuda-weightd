import pytest
import torch

from weightd.core.bundle import WeightBundle


def test_non_string_name():
    with pytest.raises(TypeError, match="must be a string"):
        WeightBundle(123, {})  # type: ignore[reportArgumentType]


def test_empty_name():
    with pytest.raises(ValueError, match="must be a non-empty string"):
        WeightBundle("", {})


def test_non_string_tensor_name():
    with pytest.raises(TypeError, match="must be a string"):
        WeightBundle("test", {123: torch.zeros(1)})  # type: ignore[reportArgumentType]


def test_empty_tensor_name():
    with pytest.raises(ValueError, match="must be a non-empty string"):
        WeightBundle("test", {"": torch.zeros(1)})


def test_non_tensor_value():
    with pytest.raises(TypeError, match="expected torch.Tensor"):
        WeightBundle("test", {"weight": [1, 2, 3]})  # type: ignore[reportArgumentType]


def test_cpu_tensor_rejected():
    with pytest.raises(ValueError, match="expected a CUDA tensor"):
        WeightBundle("test", {"weight": torch.zeros(4)})


def test_empty_bundle():
    bundle = WeightBundle("empty", {})

    assert bundle.name == "empty"
    assert len(bundle) == 0
    assert list(bundle.keys()) == []


@pytest.mark.cuda
def test_mapping_behavior(cuda_device):
    weight = torch.ones(4, 4, device=cuda_device)
    bias = torch.zeros(4, device=cuda_device)

    bundle = WeightBundle(
        "model",
        {"weight": weight, "bias": bias},
    )

    assert len(bundle) == 2
    assert set(bundle.keys()) == {"weight", "bias"}
    assert bundle["weight"] is weight
    assert bundle["bias"] is bias

    with pytest.raises(KeyError):
        _ = bundle["missing"]


@pytest.mark.cuda
def test_requires_grad_rejected(cuda_device):
    weight = torch.ones(4, device=cuda_device, requires_grad=True)

    with pytest.raises(ValueError, match="requires_grad"):
        WeightBundle("model", {"weight": weight})


@pytest.mark.cuda
def test_manifest_generation(cuda_device):
    weight = torch.arange(12, device=cuda_device, dtype=torch.float32).reshape(3, 4)

    bundle = WeightBundle("model", {"weight": weight})
    metadata = bundle.manifest.tensors["weight"]

    assert metadata.shape == (3, 4)
    assert metadata.dtype == torch.float32
    assert metadata.device == weight.device
    assert metadata.stride == (4, 1)
    assert metadata.storage_offset == 0
