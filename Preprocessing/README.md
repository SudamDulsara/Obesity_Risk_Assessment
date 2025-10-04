* project/
* ├── config/
* │   └── settings.py       # Configuration settings
* ├── modules/
* │   ├── __init__.py       # Makes modules a package
* │   ├── cleaner.py        # Data cleaning functionality
* │   ├── transformer.py    # Data transformation
* │   └── reducer.py        # Data dimentionality reduction
* │   └── validator.py      # Data validation
* ├── data/                 # Directory for input data files
* ├── output/               # Directory for processed output
* ├── main.py               # Main loading and execution script
* └── README.md            


* Gender
* Age
* Height
* Weight
* Family_history_with_overweight
* FAVC (Frequent consumption of high-caloric food)
* FCVC (Frequency of consumption of vegetables)
* NCP (Number of main meals)
* CAEC (Consumption of food between meals)
* SMOKE
* CH2O (Daily water consumption)
* SCC (Caloric beverages consumption)
* FAF (Physical activity frequency)
* TUE (Time spent using technological devices)
* CALC (Consumption of alcohol)
* MTRANS (Mode of transportation)
* 0be1dad (Target variable representing obesity level)


1. Data Loading:

    * File Format: The data is loaded from either a .csv or .xlsx file.
    * Function: load_data() handles this, selecting the appropriate method based on the file extension.

2. Data Validation:

    * Schema Validation: Checks for required columns (REQUIRED_COLUMNS), missing values (MISSING_VALUE_POLICY), duplicate rows, and other violations (e.g., enum violations, range violations).
    * ID Validation: Ensures IDs are present and checks for duplicates in the ID column.
    * Enum and Range Violations: Verifies that categorical columns (like Gender, MTRANS) only contain allowed values, and numeric columns are within predefined ranges.

3. Removing Duplicates:

    * Function: The remove_duplicates() function removes duplicate rows.
    * Action: The dataset is cleaned to ensure no repeated records.

4. Handling Missing Values:

    * Policy-Based Handling: Missing values are addressed according to the settings defined in MISSING_VALUE_POLICY.
    * Drop Rows: Certain columns (id, NObeyesdad) are excluded if they contain missing values.
    * Fill Values: Columns are filled with specific values (e.g., fill_zero, fill_unknown, fill_mean, etc.), based on the type of missingness.

5. Standardizing Columns:

    * Renaming Columns: The column names are standardized using the COLUMN_RENAME_MAP from the settings.
    * Action: Columns like NCP are renamed to more descriptive names, improving consistency.

6. Reducing Dataset:

    * Downsampling: Optional row downsampling to reduce the dataset size (ROW_SAMPLE_MAX).
    * BMI Creation: If specified (CREATE_BMI), a new BMI column is created from the height and weight columns, and the original columns are dropped.
    * Removing Near-Constant Columns: Columns where the most frequent value occurs at a high threshold (NEAR_CONSTANT_FREQ) are dropped.
    * Removing Highly Correlated Columns: Highly correlated numeric columns (above HIGH_CORR_THRESHOLD) are removed to reduce redundancy.
    * Pooling Rare Categories: Categories with very few occurrences (less than RARE_CATEGORY_MIN_COUNT) are pooled into an "Other" category to avoid sparse data issues.

7. Saving the Cleaned Data:

    * File Saving: The cleaned dataset is saved to a new CSV file in the output directory.
    * Output Path: cleaned_<filename>.csv is generated and stored.

8. Reporting:

    * Minimal Reporting: The process prints reports about what was cleaned, such as:
    * Number of duplicate rows removed.
    * Number of rows dropped due to missing values.
    * Columns renamed.
    * Any reductions applied (e.g., rows or columns dropped, BMI created).