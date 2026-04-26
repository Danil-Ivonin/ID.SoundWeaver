from app.dependencies import RuntimeQueue


def test_runtime_queue_sends_task_by_name(monkeypatch):
    enqueued = {}

    class FakeCeleryApp:
        def send_task(self, task_name, args):
            enqueued["task_name"] = task_name
            enqueued["args"] = args

    monkeypatch.setattr("app.dependencies.celery_app", FakeCeleryApp())

    RuntimeQueue().enqueue("job_1")

    assert enqueued == {"task_name": "prepare_transcription_job", "args": ["job_1"]}
