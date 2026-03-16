"""Synthetic FHIR R4 Bundle fixture."""
from __future__ import annotations


SAMPLE_FHIR_BUNDLE = {
    "resourceType": "Bundle",
    "type": "collection",
    "entry": [
        {
            "resource": {
                "resourceType": "Patient",
                "id": "patient-001",
                "birthDate": "1975-03-15",
                "gender": "male",
            }
        },
        {
            "resource": {
                "resourceType": "Condition",
                "code": {
                    "coding": [
                        {"system": "http://hl7.org/fhir/sid/icd-10", "code": "E11", "display": "Type 2 diabetes mellitus"}
                    ],
                    "text": "Type 2 diabetes mellitus",
                },
                "clinicalStatus": {
                    "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}],
                    "text": "Active",
                },
                "subject": {"reference": "Patient/patient-001"},
            }
        },
        {
            "resource": {
                "resourceType": "Observation",
                "code": {
                    "coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "HbA1c"}],
                    "text": "HbA1c",
                },
                "valueQuantity": {"value": 8.2, "unit": "%"},
                "effectiveDateTime": "2025-01-10",
                "subject": {"reference": "Patient/patient-001"},
            }
        },
        {
            "resource": {
                "resourceType": "MedicationStatement",
                "medicationCodeableConcept": {
                    "coding": [{"display": "Metformin 500mg"}],
                    "text": "Metformin",
                },
                "status": "active",
                "subject": {"reference": "Patient/patient-001"},
            }
        },
    ],
}
