from pathlib import Path

import pandas as pd

from src.data_quality.ingestion import load_csv
from src.data_quality.validators import(MissingValueValidator,PositiveValueValidator,NonNegativeValueValidator,UniqueValueValidator,StandardizationValidator)


def main():
    file_path="data/sales_data_500_rows.csv"
    df=load_csv(file_path)
    print('\nDataset Preview:')
    print(df.head())

    print('\nDataset Shape:')
    print(df.shape)

    print('\nColumn Names:')
    print(df.columns.to_list())

    print('\nDataset Information:')
    print(df.info())

    print('Sastistical Summary:')
    print(df.describe())

    print('\n' + "="*60)

    validators=[MissingValueValidator(column) for column in df.columns]
    validators.extend([PositiveValueValidator(column) for column in df.select_dtypes(include=['int']).columns])
    validators.extend([NonNegativeValueValidator(column) for column in df.select_dtypes(include=['float']).columns])
    validators.extend([UniqueValueValidator('TransactionID')])
    validators.extend([StandardizationValidator(column) for column in df.select_dtypes(include=['object','string']).columns])
    total_issues = 0
    report_rows = []
    for validator in validators:
        result=validator.validate(df)
        print(f'\nRule: {result.rule_name}')
        print(f'Column: {result.column}')
        print(f'Rows with issues: {result.issue_count}')
        if result.issue_count:
            print(f'Message: Errors found in column "{result.column}".')
        else:
            print(f'Message: No errors found in column "{result.column}".')
        if result.error_rows:
            for error_type, rows in result.error_rows.items():
                if rows:
                    if result.rule_name == 'Standardization':
                        descriptions = {
                            'empty_string': 'Empty string found at row index(es)',
                            'whitespace_only': 'Whitespace-only string found at row index(es)',
                            'leading_or_trailing_whitespace': 'Leading/trailing whitespace found at row index(es)',
                        }
                        print(f'  {descriptions[error_type]}: {rows}')
                    else:
                        print(f'  {error_type}: {len(rows)} row(s) — indexes {rows}')
                    error_descriptions = {
                        'missing_value': 'Missing value',
                        'not_numeric': 'Value is not numeric',
                        'zero_or_negative': 'Value must be greater than zero',
                        'negative_value': 'Value must be zero or greater',
                        'duplicate_value': 'Value is duplicated',
                        'empty_string': 'Empty string found',
                        'whitespace_only': 'Whitespace-only string found',
                        'leading_or_trailing_whitespace': 'Leading or trailing whitespace found',
                    }
                    for row_index in rows:
                        value = df.at[row_index, result.column]
                        if pd.isna(value):
                            value = '<missing>'
                        try:
                            csv_row_number = int(row_index) + 1
                        except (TypeError, ValueError):
                            csv_row_number = ''
                        report_rows.append({
                            'csv_row_number': csv_row_number,
                            'dataframe_index': row_index,
                            'column': result.column,
                            'rule': result.rule_name,
                            'error_type': error_type,
                            'description': error_descriptions.get(error_type, error_type),
                            'value': value,
                        })
        else:
            print('  No issues found.')
        total_issues+=result.issue_count
        print(f'Total row findings across rules so far: {total_issues}')
        print('=' * 60)

    report_path = Path('reports') / 'validation_report.csv'
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_columns = [
        'csv_row_number', 'dataframe_index', 'column', 'rule',
        'error_type', 'description', 'value',
    ]
    pd.DataFrame(report_rows, columns=report_columns).to_csv(report_path, index=False)
    print(f'Validation report saved to: {report_path.resolve()}')
              
if __name__=='__main__':
    main()
