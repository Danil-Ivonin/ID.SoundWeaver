from dataclasses import dataclass

UNKNOWN_SPEAKER = "UNKNOWN"


@dataclass(frozen=True)
class WordTimestamp:
    text: str
    start: float
    end: float


@dataclass(frozen=True)
class SpeakerSegment:
    speaker: str
    start: float
    end: float


def overlap_seconds(word: WordTimestamp, segment: SpeakerSegment) -> float:
    return max(0.0, min(word.end, segment.end) - max(word.start, segment.start))


def assign_speaker(word: WordTimestamp, segments: list[SpeakerSegment]) -> str:
    best_speaker = UNKNOWN_SPEAKER
    best_overlap = 0.0
    for segment in segments:
        overlap = overlap_seconds(word, segment)
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = segment.speaker
    return best_speaker


def build_utterances(words: list[WordTimestamp], segments: list[SpeakerSegment]) -> list[dict]:
    utterances: list[dict] = []
    for word in words:
        speaker = assign_speaker(word, segments)
        if utterances and utterances[-1]["speaker"] == speaker:
            utterances[-1]["end"] = word.end
            utterances[-1]["text"] = f"{utterances[-1]['text']} {word.text}"
            continue
        utterances.append(
            {
                "speaker": speaker,
                "start": word.start,
                "end": word.end,
                "text": word.text,
            }
        )
    return utterances
