from app.ingest import chunk_text


def test_short_text_is_one_chunk():
    assert chunk_text("Hello world.", size=100, overlap=10) == ["Hello world."]


def test_chunks_respect_size():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(30))
    chunks = chunk_text(text, size=300, overlap=50)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)


def test_overlap_carries_context():
    text = "\n\n".join(f"Sentence number {i} is here." for i in range(50))
    chunks = chunk_text(text, size=120, overlap=40)
    # the start of each chunk (after the first) repeats text from the end of the previous one
    assert any(chunks[i].split("\n\n")[0] in chunks[i - 1] for i in range(1, len(chunks)))


def test_empty_text():
    assert chunk_text("   \n\n  ", size=100, overlap=10) == []
