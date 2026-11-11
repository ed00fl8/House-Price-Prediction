from __future__ import annotations

import hashlib
import io
import zipfile

import pandas as pd
import pytest

import house_price_prediction.dataset as dataset


@pytest.fixture
def kaggle_download(
    tmp_path,
    monkeypatch,
    synthetic_dataset,
):
    csv_bytes = synthetic_dataset.to_csv(
        index=False
    ).encode("utf-8")

    archive_buffer = io.BytesIO()

    with zipfile.ZipFile(
        archive_buffer,
        "w",
        zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(
            "kc_house_data.csv",
            csv_bytes,
        )

    target = (
        tmp_path
        / "data"
        / "raw"
        / "kc_house_data.csv"
    )

    calls = []

    def fake_urlopen(request, timeout):
        calls.append(
            (request.full_url, timeout)
        )
        return io.BytesIO(archive_buffer.getvalue())

    monkeypatch.setattr(
        dataset,
        "DATASET_PATH",
        target,
    )

    monkeypatch.setattr(
        dataset,
        "EXPECTED_ROWS",
        len(synthetic_dataset),
    )

    monkeypatch.setattr(
        dataset,
        "EXPECTED_COLUMNS",
        tuple(synthetic_dataset.columns),
    )

    monkeypatch.setattr(
        dataset,
        "EXPECTED_SHA256",
        hashlib.sha256(csv_bytes).hexdigest(),
    )

    monkeypatch.setattr(
        dataset,
        "urlopen",
        fake_urlopen,
    )

    return target, calls


def test_downloads_and_validates_dataset(
    kaggle_download,
    synthetic_dataset,
):
    target, calls = kaggle_download

    source = dataset.discover_dataset()

    assert source.path == target

    pd.testing.assert_frame_equal(
        source.read(),
        synthetic_dataset,
    )

    assert target.is_file()
    assert len(calls) == 1


def test_reuses_valid_cache_without_network(
    kaggle_download,
    monkeypatch,
):
    target, calls = kaggle_download

    dataset.discover_dataset()

    def unexpected_download(*args, **kwargs):
        pytest.fail(
            "A valid cached dataset must not be downloaded again."
        )

    monkeypatch.setattr(
        dataset,
        "urlopen",
        unexpected_download,
    )

    source = dataset.discover_dataset()

    assert source.path == target
    assert len(calls) == 1


def test_replaces_modified_cached_dataset(
    kaggle_download,
):
    target, calls = kaggle_download

    target.parent.mkdir(parents=True)

    target.write_text(
        "modified data",
        encoding="utf-8",
    )

    source = dataset.discover_dataset()

    assert source.path == target
    assert len(calls) == 1

    assert (
        dataset._sha256(target.read_bytes())
        == dataset.EXPECTED_SHA256
    )


def test_rejects_unexpected_download_hash(
    kaggle_download,
    monkeypatch,
):
    _, calls = kaggle_download

    monkeypatch.setattr(
        dataset,
        "EXPECTED_SHA256",
        "0" * 64,
    )

    with pytest.raises(
        RuntimeError,
        match="pinned dataset version",
    ):
        dataset.discover_dataset()

    assert len(calls) == 1


def test_rejects_archive_without_expected_csv():
    archive_buffer = io.BytesIO()

    with zipfile.ZipFile(
        archive_buffer,
        "w",
    ) as archive:
        archive.writestr(
            "other.csv",
            "a,b\n1,2\n",
        )

    with pytest.raises(
        ValueError,
        match="Expected exactly one",
    ):
        dataset._extract_csv(
            archive_buffer.getvalue()
        )


def test_local_csv_override_for_controlled_tests(
    tmp_path,
    synthetic_dataset,
):
    path = tmp_path / "synthetic.csv"

    synthetic_dataset.to_csv(
        path,
        index=False,
    )

    source = dataset.discover_dataset(path)

    assert source.path == path.resolve()
    assert len(source.read()) == len(synthetic_dataset)


def test_duplicate_property_ids_are_allowed(
    tmp_path,
    synthetic_dataset,
):
    data = synthetic_dataset.copy()

    data.loc[1, "id"] = data.loc[0, "id"]

    path = tmp_path / "duplicate_ids.csv"

    data.to_csv(
        path,
        index=False,
    )

    source = dataset.discover_dataset(path)

    assert len(source.read()) == len(data)