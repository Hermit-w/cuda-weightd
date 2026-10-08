import select
import subprocess
import sys
import time
from pathlib import Path

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent

PRODUCER = SCRIPT_DIR / "producer.py"
CONSUMER = SCRIPT_DIR / "consumer.py"


def wait_for_output(
    process: subprocess.Popen, expected_output: str, timeout: float = 10.0
) -> None:
    """Wait for a specific output from a subprocess within a timeout."""
    if process.stdout is None:
        raise RuntimeError("Process stdout is not set")

    end_time = time.monotonic() + timeout
    buffer = ""

    while time.monotonic() < end_time:
        remaininig_time = end_time - time.monotonic()

        readable, _, _ = select.select(
            [process.stdout], [], [], min(0.5, remaininig_time)
        )
        if readable:
            line = process.stdout.readline()
            if not line:
                break  # EOF
            buffer += line
            if expected_output in buffer:
                return

    raise TimeoutError(
        f"Expected output '{expected_output}' not found in process output:\n{buffer}"
    )


@pytest.mark.cuda
@pytest.mark.integration
def test_cuda_ipc_bundle(cuda_device, socket_path):
    producer = subprocess.Popen(
        [sys.executable, "-u", str(PRODUCER), socket_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    try:
        # 等待 Producer 完成初始化
        wait_for_output(producer, "READY", timeout=10.0)

        # Consumer 是独立启动的新 Python 进程
        consumer = subprocess.run(
            [sys.executable, str(CONSUMER), socket_path],
            capture_output=True,
            text=True,
            timeout=30,
        )

        assert (
            consumer.returncode == 0
        ), f"Consumer failed: {consumer.stdout}\n{consumer.stderr}"

        assert "Consumer test passed" in consumer.stdout

        # 等待 Producer 正常结束
        stdout, stderr = producer.communicate(timeout=30)

        assert producer.returncode == 0, stderr
        assert "Producer test passed" in stdout

    finally:
        if producer.poll() is None:
            producer.terminate()

            try:
                producer.wait(timeout=5)
            except subprocess.TimeoutExpired:
                producer.kill()
                producer.wait()
