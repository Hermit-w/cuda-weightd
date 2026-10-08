import gc
import os
import sys
from multiprocessing.connection import Client

import torch

from weightd.ipc.torch_transport import recv_bundle

SOCKET_PATH = sys.argv[1] if len(sys.argv) > 1 else "/tmp/weightd-ipc-test.sock"
AUTHKEY = b"weightd-poc"


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    print(f"Consumer PID: {os.getpid()}")

    with Client(
        address=SOCKET_PATH,
        family="AF_UNIX",
        authkey=AUTHKEY,
    ) as conn:
        bundle = recv_bundle(conn)

        print(f"Bundle: {bundle.name}")
        print(f"Tensor names: {list(bundle.keys())}")

        # 只在测试环境中修改共享内存
        weight = bundle["layer.weight"]
        view = bundle["layer.view"]
        bias = bundle["layer.bias"]

        assert weight.shape == (4, 4)
        assert view.shape == (2, 4)
        assert bias.shape == (4,)

        # 验证内容和 Storage aliasing
        assert weight[0, 0].item() == 0.0

        assert (
            weight.untyped_storage().data_ptr() == view.untyped_storage().data_ptr()
        ), "Storage aliasing was not preserved"

        # 测试期间允许原地修改共享数据
        view[0, 0] = 123.0
        torch.cuda.synchronize()

        assert weight[0, 0].item() == 123.0

        conn.send("MODIFIED")

        response = conn.recv()
        assert response == "VERIFIED"

        # 清理本地 CUDA Tensor 引用
        del weight, view, bias, bundle
        gc.collect()

        conn.send("RELEASED")
        assert conn.recv() == "OK"

    print("Consumer test passed")


if __name__ == "__main__":
    main()
