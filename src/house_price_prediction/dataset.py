"""Download and validate the pinned King County dataset from Kaggle."""

from __future__ import annotations

import hashlib
import io
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from .features import RAW_FEATURE_COLUMNS, TARGET_COLUMN


DATASET_URL = (
    "https://www.kaggle.com/api/v1/datasets/download/"
    "harlfoxem/housesalesprediction"
)

DATASET_FILENAME = "kc_house_data.csv"

EXPECTED_ROWS = 21_613

EXPECTED_SHA256 = (
    "d0875baa0251b21d4bdc9d2ae940a4fe"
    "0bb6009824f23dd0e2a5b2bf04557b7e"
)

EXPECTED_COLUMNS = (
    "id",
    "date",
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
)

MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
MAX_CSV_BYTES = 10 * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = 60

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = PROJECT_ROOT / "data" / "raw" / DATASET_FILENAME

REQUIRED_COLUMNS = (
    set(RAW_FEATURE_COLUMNS)
    | {"id", "yr_renovated", TARGET_COLUMN}
)


@dataclass(frozen=True)
class DatasetSource:
    """Validated local copy of the canonical dataset."""

    path: Path

    @property
    def label(self) -> str:
        return f"Kaggle: {DATASET_URL}"

    def read(self) -> pd.DataFrame:
        try:
            return pd.read_csv(self.path, low_memory=False)
        except (
            OSError,
            UnicodeDecodeError,
            pd.errors.ParserError,
        ) as exc:
            raise ValueError(
                f"Could not read dataset {self.path}: {exc}"
            ) from exc


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _validate_frame(
    frame: pd.DataFrame,
    *,
    canonical: bool,
) -> None:
    columns = tuple(
        str(column).strip()
        for column in frame.columns
    )

    if canonical and columns != EXPECTED_COLUMNS:
        raise ValueError(
            "The Kaggle CSV schema does not match "
            "the pinned King County dataset. "
            f"Expected {list(EXPECTED_COLUMNS)}, "
            f"received {list(columns)}."
        )

    missing = sorted(
        REQUIRED_COLUMNS - set(columns)
    )

    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}"
        )

    if frame.empty:
        raise ValueError("Dataset contains no records.")

    if canonical and len(frame) != EXPECTED_ROWS:
        raise ValueError(
            f"Unexpected number of records: "
            f"expected {EXPECTED_ROWS:,}, "
            f"received {len(frame):,}."
        )

    if frame["id"].isna().any():
        raise ValueError(
            "Dataset contains missing property IDs."
        )

    # Repeated property IDs are valid in this
    # historical house-sales dataset.
    prices = pd.to_numeric(
        frame[TARGET_COLUMN],
        errors="coerce",
    )

    if prices.isna().any() or (prices <= 0).any():
        raise ValueError(
            "Dataset contains missing or non-positive sale prices."
        )


def _extract_csv(archive_bytes: bytes) -> bytes:
    try:
        with zipfile.ZipFile(
            io.BytesIO(archive_bytes)
        ) as archive:

            matches = [
                item
                for item in archive.infolist()
                if (
                    not item.is_dir()
                    and item.filename.replace("\\", "/")
                    .rsplit("/", 1)[-1].lower()
                    == DATASET_FILENAME
                )
            ]

            if len(matches) != 1:
                raise ValueError(
                    f"Expected exactly one {DATASET_FILENAME} "
                    f"in the Kaggle archive; found {len(matches)}."
                )

            info = matches[0]

            if info.file_size > MAX_CSV_BYTES:
                raise ValueError(
                    "The extracted CSV exceeds the 10 MB safety limit."
                )

            return archive.read(info)

    except zipfile.BadZipFile as exc:
        raise ValueError(
            "Kaggle did not return a valid ZIP archive."
        ) from exc


def _download_csv() -> bytes:
    request = Request(
        DATASET_URL,
        headers={
            "User-Agent": (
                "king-county-house-price-prediction/1.0"
            )
        },
    )

    try:
        with urlopen(
            request,
            timeout=DOWNLOAD_TIMEOUT_SECONDS,
        ) as response:
            archive_bytes = response.read(
                MAX_ARCHIVE_BYTES + 1
            )

        if len(archive_bytes) > MAX_ARCHIVE_BYTES:
            raise ValueError(
                "The Kaggle archive exceeds the 20 MB download limit."
            )

        csv_bytes = _extract_csv(archive_bytes)

        actual_hash = _sha256(csv_bytes)

        if actual_hash != EXPECTED_SHA256:
            raise ValueError(
                "The downloaded CSV does not match "
                "the pinned dataset version. "
                f"Expected SHA-256: {EXPECTED_SHA256}. "
                f"Received: {actual_hash}. "
                "No model was trained."
            )

        frame = pd.read_csv(
            io.BytesIO(csv_bytes),
            low_memory=False,
        )

        _validate_frame(
            frame,
            canonical=True,
        )

        return csv_bytes

    except (
        HTTPError,
        URLError,
        TimeoutError,
        OSError,
        zipfile.BadZipFile,
        ValueError,
    ) as exc:
        raise RuntimeError(
            "Could not download or validate the "
            "pinned King County dataset from Kaggle. "
            "Check the internet connection. "
            f"Source: {DATASET_URL}. Details: {exc}"
        ) from exc


def _cached_file_is_valid() -> bool:
    if not DATASET_PATH.is_file():
        return False

    try:
        content = DATASET_PATH.read_bytes()

        if _sha256(content) != EXPECTED_SHA256:
            return False

        frame = pd.read_csv(
            io.BytesIO(content),
            low_memory=False,
        )

        _validate_frame(
            frame,
            canonical=True,
        )

        return True

    except (
        OSError,
        UnicodeDecodeError,
        pd.errors.ParserError,
        ValueError,
    ):
        return False


def _install_csv(csv_bytes: bytes) -> None:
    DATASET_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".csv",
            prefix="kc-house-",
            dir=DATASET_PATH.parent,
            delete=False,
        ) as temporary:

            temporary.write(csv_bytes)
            temporary_path = Path(temporary.name)

        temporary_path.replace(DATASET_PATH)

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def discover_dataset(
    data_path: str | Path | None = None,
) -> DatasetSource:
    """Return the pinned Kaggle dataset, downloading it if necessary.

    data_path is retained for automated tests and controlled
    local experiments. Normal execution downloads the dataset
    automatically and requires no manual file selection.
    """

    if data_path is not None:
        path = Path(data_path).expanduser().resolve()

        if (
            not path.is_file()
            or path.suffix.lower() != ".csv"
        ):
            raise FileNotFoundError(
                f"Dataset CSV not found: {path}"
            )

        frame = pd.read_csv(
            path,
            low_memory=False,
        )

        _validate_frame(
            frame,
            canonical=False,
        )

        return DatasetSource(path)

    if not _cached_file_is_valid():
        csv_bytes = _download_csv()
        _install_csv(csv_bytes)

    return DatasetSource(DATASET_PATH)