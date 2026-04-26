from pathlib import Path


def test_dockerfile_targets_python_312_and_cuda_128():
    dockerfile = Path("Dockerfile").read_text()

    assert "FROM python:3.12-slim" in dockerfile
    assert "AS api" in dockerfile
    assert "FROM nvidia/cuda:12.8" in dockerfile
    assert "AS worker" in dockerfile
    assert "python3.12" in dockerfile
    assert "uv.lock" in dockerfile
    assert "uv sync --frozen" in dockerfile
    assert "--extra worker" in dockerfile
    assert "python3.12 -m pip install --no-cache-dir uv" not in dockerfile
    assert 'pip install ".[dev]"' not in dockerfile
    assert "python3.10" not in dockerfile
    assert "libpython3.12" in dockerfile
