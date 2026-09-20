import pytest

from rag_qa.ingest import load_documents


def test_loads_txt_and_md_files(tmp_path):
    (tmp_path / "a.txt").write_text("hello from a")
    (tmp_path / "b.md").write_text("# hello from b")
    (tmp_path / "ignore.pdf").write_text("should be skipped")

    docs = load_documents(tmp_path)

    assert {d.doc_id for d in docs} == {"a.txt", "b.md"}
    assert {d.text for d in docs} == {"hello from a", "# hello from b"}


def test_walks_subdirectories(tmp_path):
    sub = tmp_path / "nested"
    sub.mkdir()
    (sub / "c.txt").write_text("nested content")

    docs = load_documents(tmp_path)

    assert docs[0].doc_id == "nested/c.txt"


def test_skips_empty_files(tmp_path):
    (tmp_path / "empty.txt").write_text("   ")
    (tmp_path / "real.txt").write_text("content")

    docs = load_documents(tmp_path)

    assert len(docs) == 1
    assert docs[0].doc_id == "real.txt"


def test_missing_source_dir_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(tmp_path / "does-not-exist")
