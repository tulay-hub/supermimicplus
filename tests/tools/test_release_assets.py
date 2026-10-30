import hashlib
import io
import json
import sys
import tarfile

import pytest
from tools import fetch_release_assets as fetch


def prepared(tmp_path, monkeypatch, name="payload/value.txt", content=b"verified"):
    root = tmp_path / "project"
    cache = root / "release-local/downloads"
    cache.mkdir(parents=True)
    archive = cache / "sample.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        info = tarfile.TarInfo(name)
        info.size = len(content)
        stream.addfile(info, io.BytesIO(content))
    manifest = {
        "assets": [
            {
                "name": archive.name,
                "kind": "data",
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        ]
    }
    (root / "release-manifest.json").write_text(json.dumps(manifest))
    monkeypatch.setattr(fetch, "__file__", str(root / "tools/fetch_release_assets.py"))
    monkeypatch.setattr(sys, "argv", ["fetch_release_assets.py"])
    return root


def test_verified_install_is_idempotent(tmp_path, monkeypatch):
    root = prepared(tmp_path, monkeypatch)
    fetch.main()
    fetch.main()
    assert (root / "payload/value.txt").read_bytes() == b"verified"


def test_changed_local_file_is_preserved(tmp_path, monkeypatch):
    root = prepared(tmp_path, monkeypatch)
    (root / "payload").mkdir()
    (root / "payload/value.txt").write_bytes(b"local change")
    with pytest.raises(FileExistsError):
        fetch.main()
    assert (root / "payload/value.txt").read_bytes() == b"local change"


def test_parent_traversal_is_rejected(tmp_path, monkeypatch):
    prepared(tmp_path, monkeypatch, name="../outside")
    with pytest.raises(ValueError, match="Unsafe"):
        fetch.main()
    assert not (tmp_path / "outside").exists()


def test_checksum_mismatch_does_not_extract(tmp_path, monkeypatch):
    root = prepared(tmp_path, monkeypatch)
    path = root / "release-manifest.json"
    data = json.loads(path.read_text())
    data["assets"][0]["sha256"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Checksum"):
        fetch.main()
    assert not (root / "payload").exists()
