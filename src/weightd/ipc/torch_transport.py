from multiprocessing.connection import Connection

import torch
import torch.multiprocessing  # 注册 PyTorch Tensor reducers

from weightd.core.bundle import WeightBundle
from weightd.core.manifest import BundleManifest


def send_tensor(
    conn: Connection,
    tensor: torch.Tensor,
) -> None:
    if not tensor.is_cuda:
        raise ValueError("Expected a CUDA tensor")

    conn.send(tensor)


def recv_tensor(conn: Connection) -> torch.Tensor:
    tensor = conn.recv()

    if not isinstance(tensor, torch.Tensor):
        raise TypeError("Received object is not a Tensor")

    if not tensor.is_cuda:
        raise ValueError("Received Tensor is not on CUDA")

    return tensor


def pack_bundle(
    bundle: WeightBundle,
) -> tuple[BundleManifest, dict[str, torch.Tensor]]:
    """
    Pack a WeightBundle into a manifest and a dictionary of tensors.
    """
    return (bundle.manifest, dict(bundle.items()))


def unpack_bundle(
    payload: tuple[BundleManifest, dict[str, torch.Tensor]],
) -> WeightBundle:
    """
    Unpack a manifest and a dictionary of tensors into a WeightBundle.
    """
    manifest, tensors = payload

    if not isinstance(manifest, BundleManifest):
        raise TypeError("Received object is not a BundleManifest")

    if not isinstance(tensors, dict):
        raise TypeError("Received object is not a dict of tensors")

    bundle = WeightBundle(
        name=manifest.name,
        tensors=tensors,
    )

    if bundle.manifest != manifest:
        raise ValueError("Received bundle does not match the manifest")

    return bundle


def send_bundle(
    conn: Connection,
    bundle: WeightBundle,
) -> None:
    conn.send(pack_bundle(bundle))


def recv_bundle(conn: Connection) -> WeightBundle:
    return unpack_bundle(conn.recv())
