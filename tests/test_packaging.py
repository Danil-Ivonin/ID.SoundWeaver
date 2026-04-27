from pathlib import Path

import tomllib
import yaml


def test_pyproject_splits_worker_dependencies() -> None:
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())

    dependencies = set(pyproject["project"]["dependencies"])
    optional = pyproject["project"]["optional-dependencies"]
    worker = set(optional["worker"])

    assert "fastapi>=0.136.0" in dependencies
    assert "torch==2.8.0" not in dependencies
    assert "torchaudio==2.8.0" not in dependencies
    assert "pyannote-audio>=4.0.4" not in dependencies

    assert "torch==2.8.0" in worker
    assert "torchaudio==2.8.0" in worker
    assert "pyannote-audio>=4.0.4" in worker
    assert "gigaam" in worker


def test_compose_uses_separate_api_and_worker_targets() -> None:
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())

    assert compose["services"]["api"]["build"]["target"] == "api"
    assert compose["services"]["worker"]["build"]["target"] == "worker"

    api_command = compose["services"]["api"]["command"]
    assert "uvicorn app.main:app" in api_command

    worker_command = compose["services"]["worker"]["command"]
    assert "celery -A app.tasks.celery_app.celery_app worker" in worker_command
    assert "--concurrency=1" in worker_command


def test_compose_postgres_uses_official_env_and_worker_is_gpu_profile() -> None:
    compose = yaml.safe_load(Path("docker-compose.yml").read_text())

    postgres = compose["services"]["postgres"]
    assert postgres["image"] == "postgres:16"
    assert postgres["environment"] == {
        "POSTGRES_USER": "${DB_USER:-soundweaver}",
        "POSTGRES_PASSWORD": "${DB_PASSWORD:-soundweaver}",
        "POSTGRES_DB": "${DB_NAME:-soundweaver}",
    }
    assert "command" not in postgres

    worker = compose["services"]["worker"]
    assert worker["profiles"] == ["gpu"]
    assert "TRANSFORMERS_CACHE" not in worker["environment"]
