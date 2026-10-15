from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ValidationResult:
    """Result from one validation rule, grouped by error type."""

    rule_name: str
    column: str
    invalid_rows: list[int]
    message: str
    error_rows: dict[str, list[int]]=field(default_factory=dict)

    @property
    def issue_count(self) -> int:
        """Number of distinct rows with at least one issue for this rule."""
        return len(self.invalid_rows)

    @property
    def error_counts(self) -> dict[str, int]:
        """Number of affected rows for each error type."""
        return {error_type: len(rows) for error_type, rows in self.error_rows.items()}


def _make_result(
    rule_name: str,
    column: str,
    message: str,
    error_rows: dict[str, list[int]],
) -> ValidationResult:
    # A row can have more than one error type, so count distinct row indexes.
    invalid_rows = list(dict.fromkeys(row for rows in error_rows.values() for row in rows))
    return ValidationResult(rule_name, column, invalid_rows, message, error_rows)


def _require_column(df: pd.DataFrame, column: str) -> None:
    if column not in df.columns:
        raise ValueError(f"Column not found: {column}")


class BaseValidator(ABC):
    @abstractmethod
    def validate(self, df: pd.DataFrame) -> ValidationResult:
        raise NotImplementedError


class MissingValueValidator(BaseValidator):
    def __init__(self, column: str):
        self.column = column

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        _require_column(df, self.column)
        missing_rows = df.index[df[self.column].isna()].tolist()
        return _make_result(
            "Missing_Values", self.column,
            f"Found {len(missing_rows)} missing value(s) in {self.column}.",
            {"missing_value": missing_rows},
        )


class PositiveValueValidator(BaseValidator):
    def __init__(self, column: str):
        self.column = column

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        _require_column(df, self.column)
        original = df[self.column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid_type = (original.notna() & numeric.isna()).fillna(False)
        non_positive = (numeric.notna() & numeric.le(0)).fillna(False)
        errors = {
            "not_numeric": df.index[invalid_type].tolist(),
            "zero_or_negative": df.index[non_positive].tolist(),
        }
        return _make_result(
            "Positive_Values", self.column,
            f"Expected numeric values greater than zero in {self.column}.", errors,
        )


class NonNegativeValueValidator(BaseValidator):
    def __init__(self, column: str):
        self.column = column

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        _require_column(df, self.column)
        original = df[self.column]
        numeric = pd.to_numeric(original, errors="coerce")
        invalid_type = (original.notna() & numeric.isna()).fillna(False)
        negative = (numeric.notna() & numeric.lt(0)).fillna(False)
        errors = {
            "not_numeric": df.index[invalid_type].tolist(),
            "negative_value": df.index[negative].tolist(),
        }
        return _make_result(
            "NonNegative_Values", self.column,
            f"Expected numeric values greater than or equal to zero in {self.column}.", errors,
        )


class UniqueValueValidator(BaseValidator):
    def __init__(self, column: str):
        self.column = column

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        _require_column(df, self.column)
        # Missing values are reported by MissingValueValidator, not as duplicates.
        duplicate_mask = df[self.column].notna() & df[self.column].duplicated(keep="first")
        duplicate_rows = df.index[duplicate_mask].tolist()
        return _make_result(
            "Unique_Values", self.column,
            f"Found {len(duplicate_rows)} duplicate value(s) in {self.column}.",
            {"duplicate_value": duplicate_rows},
        )


class StandardizationValidator(BaseValidator):
    def __init__(self, column: str):
        self.column = column

    def validate(self, df: pd.DataFrame) -> ValidationResult:
        _require_column(df, self.column)
        values = df[self.column].astype("string")
        non_missing = values.notna()
        empty_string = (non_missing & values.eq("")).fillna(False)
        whitespace_only = (
            non_missing & values.ne("") & values.str.strip().eq("")
        ).fillna(False)
        leading_or_trailing = (
            non_missing & values.str.strip().ne("") & values.ne(values.str.strip())
        ).fillna(False)
        errors = {
            "empty_string": df.index[empty_string].tolist(),
            "whitespace_only": df.index[whitespace_only].tolist(),
            "leading_or_trailing_whitespace": df.index[leading_or_trailing].tolist(),
        }
        return _make_result(
            "Standardization", self.column,
            f"Found whitespace standardization issue(s) in {self.column}.", errors,
        )
