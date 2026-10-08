from collections.abc import Iterable, Mapping

import torch

from weightd.core.manifest import BundleManifest, TensorMetadata


class WeightBundle(Mapping[str, torch.Tensor]):
    """
    This class represents a bundle of named CUDA tensors, along with their metadata.
    It provides a convenient way to manage and access multiple tensors as a single entity.
    """

    def __init__(
        self,
        name: str,
        tensors: Mapping[str, torch.Tensor],
    ) -> None:

        if not isinstance(name, str):
            raise TypeError(f"Bundle name must be a string, got {type(name).__name__}")

        if not name:
            raise ValueError("Bundle name must be a non-empty string.")

        self.name = name
        self._tensors = dict(tensors)

        for key, tensor in self._tensors.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Tensor name must be a string, got {type(key).__name__}"
                )

            if not key:
                raise ValueError("Tensor name must be a non-empty string.")

            if not isinstance(tensor, torch.Tensor):
                raise TypeError(
                    f"{key}: expected torch.Tensor, got {type(tensor).__name__}"
                )

            if not tensor.is_cuda:
                raise ValueError(f"{key}: expected a CUDA tensor, got {tensor.device}")

            if tensor.layout != torch.strided:
                # we only support strided tensors for now, as they are the most common and widely used tensor type in PyTorch.
                # Other layouts, such as sparse or quantized tensors, may require additional handling and are not supported in this implementation.
                raise ValueError(
                    f"{key}: expected a strided tensor, got {tensor.layout}"
                )

            if tensor.requires_grad:
                # the tool itself is designed only for inference, so we don't support tensors that require gradients.
                raise ValueError(
                    f"{key}: expected a tensor that does not require gradients, got requires_grad=True"
                )

    def __getitem__(self, key: str) -> torch.Tensor:
        return self._tensors[key]

    def __iter__(self) -> Iterable[str]:
        return iter(self._tensors)

    def __len__(self) -> int:
        return len(self._tensors)

    @property
    def manifest(self) -> BundleManifest:
        return BundleManifest(
            name=self.name,
            tensors={
                key: TensorMetadata.from_tensor(tensor)
                for key, tensor in self._tensors.items()
            },
        )
