"""resolve_source_directory stores the canonical root-relative form."""

from microProfiler.profiling import resolve_source_directory


def test_resolve_source_directory_relative_to_root(tmp_path):
    root = tmp_path / "ds"
    sub = root / "images" / "wellA"
    sub.mkdir(parents=True)
    f = sub / "r01c01f01p01-ch1.tiff"
    f.write_bytes(b"x")

    assert resolve_source_directory({"ch1": str(f)}, ["ch1"], root) == "images/wellA"

    root_file = root / "r01c01f01p01-ch1.tiff"
    root_file.write_bytes(b"x")
    assert resolve_source_directory({"__file__": str(root_file)}, ["ch1"], root) == "."

    assert resolve_source_directory({"ch1": None}, ["ch1"], root) == ""
