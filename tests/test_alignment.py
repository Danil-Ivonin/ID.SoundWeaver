from app.alignment import SpeakerSegment, WordTimestamp, build_utterances


def test_build_utterances_assigns_words_by_max_overlap():
    words = [
        WordTimestamp(text="hello", start=0.0, end=0.5),
        WordTimestamp(text="world", start=0.5, end=1.0),
        WordTimestamp(text="again", start=1.2, end=1.6),
    ]
    segments = [
        SpeakerSegment(speaker="SPEAKER_00", start=0.0, end=1.0),
        SpeakerSegment(speaker="SPEAKER_01", start=1.1, end=2.0),
    ]

    utterances = build_utterances(words, segments)

    assert utterances == [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 1.0, "text": "hello world"},
        {"speaker": "SPEAKER_01", "start": 1.2, "end": 1.6, "text": "again"},
    ]


def test_build_utterances_keeps_unmatched_words_as_unknown():
    words = [
        WordTimestamp(text="lost", start=3.0, end=3.5),
    ]

    utterances = build_utterances(words, [])

    assert utterances == [
        {"speaker": "UNKNOWN", "start": 3.0, "end": 3.5, "text": "lost"},
    ]
