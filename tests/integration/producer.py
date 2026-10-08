import os
import sys
from multiprocessing.connection import Listener
from pathlib import Path

import torch

from weightd.core.bundle import WeightBundle
from weightd.ipc.torch_transport import send_bundle

SOCKET_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/weightd-ipc-test.sock"
AUTHKEY = b"weightd-poc"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    # 创建约 64 MiB 的测试 Tensor
    base = torch.arange(16, device="cuda:0", dtype=torch.float32).reshape(4, 4)
    bundle = WeightBundle(
        name="test-model",
        tensors={
            "layer.weight": base,
            "layer.view": base[:2, :],
            "layer.bias": torch.ones(4, device="cuda:0"),
        },
    )

    socket = Path(SOCKET_PATH)
    if socket.exists():
        raise RuntimeError(
            f"Socket already exists: {SOCKET_PATH}. "
            "Ensure no other producer is running."
        )

    # print(f"Producer PID: {os.getpid()}")

    try:
        with Listener(
            address=SOCKET_PATH,
            family="AF_UNIX",
            authkey=AUTHKEY,
        ) as listener:
            print("READY", flush=True)

            with listener.accept() as conn:
                send_bundle(conn, bundle)
                print("Bundle sent", flush=True)

                # Consumer 修改共享 Tensor 后通知 Producer
                message = conn.recv()
                assert message == "MODIFIED"

                torch.cuda.synchronize()
                value = bundle["layer.weight"][0, 0].item()
                print(f"Producer sees: {value}")

                assert value == 123.0

                conn.send("VERIFIED")

                assert conn.recv() == "RELEASED"
                conn.send("OK")

        print("Producer test passed")

    finally:
        socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
