"""Synthetic ClinicalTrials.gov v2 API response fixtures."""

from __future__ import annotations

SAMPLE_STUDY_RAW = {
    "protocolSection": {
        "identificationModule": {
            "nctId": "NCT99999999",
            "briefTitle": "Study of Metformin in Type 2 Diabetes",
        },
        "statusModule": {
            "overallStatus": "RECRUITING",
            "lastUpdatePostDateStruct": {"date": "2025-01-15"},
            "startDateStruct": {"date": "2024-06-01"},
        },
        "descriptionModule": {"briefSummary": "A randomized trial evaluating metformin efficacy in adults with T2DM."},
        "conditionsModule": {"conditions": ["Type 2 Diabetes Mellitus"]},
        "armsInterventionsModule": {"interventions": [{"name": "Metformin"}, {"name": "Placebo"}]},
        "eligibilityModule": {
            "eligibilityCriteria": (
                "Inclusion Criteria:\n"
                "- Adults 18 years or older\n"
                "- Confirmed diagnosis of type 2 diabetes mellitus\n"
                "- HbA1c between 7.0 and 10.0%\n\n"
                "Exclusion Criteria:\n"
                "- Renal impairment (eGFR < 30 mL/min)\n"
                "- Current use of insulin\n"
                "- Pregnant or breastfeeding"
            ),
            "sex": "ALL",
            "minimumAge": "18 Years",
            "maximumAge": "75 Years",
        },
        "sponsorCollaboratorsModule": {"leadSponsor": {"name": "University Medical Center"}},
        "contactsLocationsModule": {"locations": [{"facility": "University Hospital"}]},
        "designModule": {"phases": ["PHASE3"]},
    }
}

SAMPLE_API_RESPONSE = {
    "studies": [SAMPLE_STUDY_RAW],
    "nextPageToken": None,
}
