from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TensorMetadata:
    shape: tuple[int, ...]
    dtype: torch.dtype
    stride: tuple[int, ...]
    storage_offset: int
    device: torch.device

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "TensorMetadata":
        return cls(
            shape=tensor.shape,
            dtype=tensor.dtype,
            stride=tensor.stride(),
            # A tensor used here should be instantiated with allocated memory,
            # so storage_offset should be an integer rather than `SymInt`.
            # TODO: Maybe we should check if the tensor is a `SymInt` and raise an error if it is.
            storage_offset=int(tensor.storage_offset()),
            device=tensor.device,
        )

    def match_tensor(self, tensor: torch.Tensor) -> bool:
        return self == TensorMetadata.from_tensor(tensor)


@dataclass(frozen=True)
class BundleManifest:
    """
    A manifest that describes a bundle of named CUDA tensors, including their metadata.
    """

    name: str
    tensors: dict[str, TensorMetadata]
