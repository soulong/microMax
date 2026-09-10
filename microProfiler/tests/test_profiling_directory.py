"""resolve_source_directory stores the DB `directory` contract value:
an absolute path with forward slashes (microBase db_contracts)."""

from microProfiler.profiling import resolve_source_directory


def test_resolve_source_directory_absolute_forward_slashes(tmp_path):
    root = tmp_path / "ds"
    sub = root / "images" / "wellA"
    sub.mkdir(parents=True)
    f = sub / "r01c01f01p01-ch1.tiff"
    f.write_bytes(b"x")

    expected = str(f.parent.resolve()).replace("\\", "/")
    assert resolve_source_directory({"ch1": str(f)}, ["ch1"], root) == expected

    # A relative channel path is anchored at the dataset root first.
    assert (resolve_source_directory(
        {"ch1": "images/wellA/r01c01f01p01-ch1.tiff"}, ["ch1"], root)
        == expected)

    # Root-level files still resolve (to the root itself, absolute form).
    root_file = root / "r01c01f01p01-ch1.tiff"
    root_file.write_bytes(b"x")
    assert (resolve_source_directory({"__file__": str(root_file)}, ["ch1"], root)
            == str(root.resolve()).replace("\\", "/"))

    assert resolve_source_directory({"ch1": None}, ["ch1"], root) == ""
