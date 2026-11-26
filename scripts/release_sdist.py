"""Build the PEP 517 source archive without recording local account details."""

import tarfile
from pathlib import Path

from setuptools.command.sdist import sdist


def _clear_owner(member):
    member.uid = member.gid = 0
    member.uname = member.gname = ""
    return member


class PublicSdist(sdist):
    def make_archive(self, base_name, format, root_dir=None, base_dir=None, owner=None, group=None):
        if format != "gztar":
            return super().make_archive(base_name, format, root_dir, base_dir, owner, group)

        archive_path = Path(f"{base_name}.tar.gz")
        archive_path.parent.mkdir(parents=True, exist_ok=True)
        tree = Path(root_dir or ".") / (base_dir or ".")
        with tarfile.open(archive_path, "w:gz", format=tarfile.PAX_FORMAT) as archive:
            archive.add(tree, arcname=base_dir or ".", filter=_clear_owner)
        return str(archive_path)
