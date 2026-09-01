# SQL Execution Order

Before running these scripts, ensure that the anonymized source tables and the two CSV sources have been loaded.

Run the construction and validation scripts in this order:

1. `01_build_base_cohort.sql`
2. `02_validate_base_cohort.sql`
3. `03_build_vital_signs_cohort.sql`
4. `04_validate_vital_signs_cohort.sql`
5. `05_build_laboratory_records.sql`
6. `06_validate_laboratory_records.sql`
7. `07_build_pharmacy_features.sql`
8. `08_validate_pharmacy_features.sql`
9. `09_build_daily_sepsis_model.sql`
10. `10_validate_daily_sepsis_model.sql`

Final table for the Python EDA and modelling pipeline:

`daily_sepsis_model`
