import os
import tarfile

import pytest
from scripts.check_release_dist import check_archive_owners
from scripts.release_sdist import PublicSdist
from setuptools import Distribution


def test_source_archive_preserves_files_modes_and_links(tmp_path):
    tree = tmp_path / "source"
    tree.mkdir()
    script = tree / "entrypoint"
    script.write_bytes(b"example\n")
    script.chmod(0o755)
    (tree / "symbolic").symlink_to(script.name)
    os.link(script, tree / "hardlink")
    long_name = "example" * 20
    (tree / long_name).touch()
    path = PublicSdist(Distribution()).make_archive(
        tmp_path / "dist" / "example", "gztar", root_dir=tmp_path, base_dir="source"
    )
    with tarfile.open(path) as archive:
        check_archive_owners(archive)
        assert archive.getmember("source").isdir()
        assert archive.getmember("source/entrypoint").mode & 0o777 == 0o755
        assert archive.getmember("source/hardlink").islnk()
        assert archive.getmember("source/symbolic").issym()
        assert archive.getmember("source/symbolic").linkname == "entrypoint"
        assert archive.getmember(f"source/{long_name}").isfile()
        for name in ("entrypoint", "hardlink", "symbolic"):
            assert archive.extractfile(f"source/{name}").read() == b"example\n"


@pytest.mark.parametrize(
    ("field", "value", "header"),
    [
        ("uid", 123, "member"),
        ("gid", 456, "member"),
        ("uname", "builder", "member"),
        ("gname", "builders", "member"),
        ("uid", "0", "pax"),
        ("uname", "builder", "global"),
    ],
)
def test_checker_rejects_ownership_even_on_directories(tmp_path, field, value, header):
    path = tmp_path / "example.tar.gz"
    with tarfile.open(path, "w:gz", pax_headers={field: value} if header == "global" else None) as archive:
        member = tarfile.TarInfo("source")
        member.type = tarfile.DIRTYPE
        if header == "member":
            setattr(member, field, value)
        elif header == "pax":
            member.pax_headers[field] = value
        archive.addfile(member)
    with tarfile.open(path) as archive, pytest.raises(ValueError, match="ownership metadata"):
        check_archive_owners(archive)
