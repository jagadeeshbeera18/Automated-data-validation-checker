from pathlib import Path
import pandas as pd
import pandas.testing as pdt
import pytest

from src.data_quality.ingestion import load_csv


def test_load_csv_returns_dataframe_with_csv_contents(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("TransactionID,Amount\n101,25.5\n102,40.0\n", encoding="utf-8")

    result = load_csv(csv_path)

    expected = pd.DataFrame(
        {"TransactionID": [101, 102], "Amount": [25.5, 40.0]}
    )
    assert isinstance(result, pd.DataFrame)
    pdt.assert_frame_equal(result, expected)


def test_load_csv_raises_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError, match="File not found"):
        load_csv(missing_path)


def test_load_csv_raises_for_non_csv_file(tmp_path: Path) -> None:
    text_path = tmp_path / "data.txt"
    text_path.write_text("not a csv", encoding="utf-8")

    with pytest.raises(ValueError, match="Expected as csv file"):
        load_csv(text_path)


def test_load_csv_raises_when_path_is_a_directory(tmp_path: Path) -> None:
    directory_path = tmp_path / "folder.csv"
    directory_path.mkdir()

    with pytest.raises(ValueError, match="Expected a file received"):
        load_csv(directory_path)
