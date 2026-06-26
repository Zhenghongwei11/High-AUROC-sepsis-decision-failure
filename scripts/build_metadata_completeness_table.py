from __future__ import annotations

import csv
from pathlib import Path


OUTPUT_PATH = Path(
    "/Users/simanan/Desktop/plos one 2/results/tables/metadata_completeness_summary.tsv"
)


ROWS = [
    {
        "dataset_id": "GSE65682",
        "age": "yes",
        "sex": "yes",
        "infection_or_clinical_context": "partial",
        "severity_or_outcome": "partial",
        "timepoint_or_followup": "partial",
        "treatment_or_exposure": "no",
        "clinical_baseline_table_feasible": "partial_only",
        "source_fields": (
            "characteristics_ch1;characteristics_ch1_2;characteristics_ch1_3;"
            "characteristics_ch1_7;characteristics_ch1_8;characteristics_ch1_9;"
            "characteristics_ch1_10;characteristics_ch1_12"
        ),
        "notes": (
            "Age and sex are present. Clinical context is available through pneumonia, "
            "ICU-acquired infection, and abdominal-sepsis/control fields, but these are "
            "heterogeneous surrogate descriptors rather than a uniform manuscript-ready "
            "infection-source table across cohorts. Mortality and time-to-event are present, "
            "but harmonized organ-dysfunction severity variables are not."
        ),
    },
    {
        "dataset_id": "GSE95233",
        "age": "yes",
        "sex": "yes",
        "infection_or_clinical_context": "no",
        "severity_or_outcome": "partial",
        "timepoint_or_followup": "yes",
        "treatment_or_exposure": "no",
        "clinical_baseline_table_feasible": "partial_only",
        "source_fields": (
            "characteristics_ch1;characteristics_ch1_2;characteristics_ch1_3;"
            "characteristics_ch1_4"
        ),
        "notes": (
            "Age, sex, time point, and survival are present. A uniform infection-source "
            "field is not available in the processed GEO metadata."
        ),
    },
    {
        "dataset_id": "GSE154918",
        "age": "no",
        "sex": "yes",
        "infection_or_clinical_context": "partial",
        "severity_or_outcome": "partial",
        "timepoint_or_followup": "yes",
        "treatment_or_exposure": "no",
        "clinical_baseline_table_feasible": "partial_only",
        "source_fields": "characteristics_ch1;characteristics_ch1_2",
        "notes": (
            "Processed GEO metadata provide sex and status labels such as healthy, infection, "
            "sepsis, shock, and follow-up, but no harmonizable age field and no detailed "
            "treatment or infection-site variables."
        ),
    },
    {
        "dataset_id": "GSE28750",
        "age": "no",
        "sex": "no",
        "infection_or_clinical_context": "partial",
        "severity_or_outcome": "no",
        "timepoint_or_followup": "partial",
        "treatment_or_exposure": "no",
        "clinical_baseline_table_feasible": "partial_only",
        "source_fields": "characteristics_ch1_2;description",
        "notes": (
            "Health status and post-surgical context are recoverable, and the free-text "
            "description notes early enrollment and culture-positive sepsis, but consistent "
            "age, sex, and severity fields are not available."
        ),
    },
]


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ROWS[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(ROWS)


if __name__ == "__main__":
    main()
