from app.alignment import SpeakerSegment


class PyannoteDiarizationService:
    def __init__(self, model_name: str, token: str, device: str) -> None:
        import torch
        from pyannote.audio import Pipeline

        self.pipeline = Pipeline.from_pretrained(model_name, token=token)
        self.pipeline.to(torch.device(device))

    def diarize(
        self,
        *,
        waveform,
        sample_rate: int,
        num_speakers: int | None,
        min_speakers: int | None,
        max_speakers: int | None,
    ) -> list[SpeakerSegment]:
        kwargs = {
            "num_speakers": num_speakers,
            "min_speakers": min_speakers,
            "max_speakers": max_speakers,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        output = self.pipeline({"waveform": waveform, "sample_rate": sample_rate}, **kwargs)
        return [
            SpeakerSegment(speaker=speaker, start=float(turn.start), end=float(turn.end))
            for turn, speaker in output.speaker_diarization
        ]
