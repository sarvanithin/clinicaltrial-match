"""
Async ClinicalTrials.gov API v2 client with pagination.

Fetches trials by condition query, handles pagination via pageToken,
and normalizes the deeply-nested v2 response into Trial dicts.
"""
from __future__ import annotations

import re
import time
from typing import Any, AsyncIterator

import httpx
import structlog
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from clinicaltrial_match.config import TrialsConfig
from clinicaltrial_match.trials.models import Trial, TrialStatus

_logger = structlog.get_logger()


_FIELDS = ",".join([
    "NCTId", "BriefTitle", "BriefSummary", "Condition", "InterventionName",
    "Phase", "OverallStatus", "EligibilityCriteria", "Sex", "MinimumAge", "MaximumAge",
    "LeadSponsorName", "LocationFacility", "StartDate",
    "PrimaryCompletionDate", "LastUpdatePostDate",
])

_STATUS_MAP: dict[str, TrialStatus] = {
    "RECRUITING": TrialStatus.RECRUITING,
    "NOT_YET_RECRUITING": TrialStatus.NOT_YET_RECRUITING,
    "ACTIVE_NOT_RECRUITING": TrialStatus.ACTIVE_NOT_RECRUITING,
    "COMPLETED": TrialStatus.COMPLETED,
    "TERMINATED": TrialStatus.TERMINATED,
    "WITHDRAWN": TrialStatus.WITHDRAWN,
    "SUSPENDED": TrialStatus.SUSPENDED,
}


def _parse_age_string(age_str: str | None) -> float | None:
    """Convert '18 Years' → 18.0, '6 Months' → 0.5, None → None."""
    if not age_str:
        return None
    match = re.match(r"(\d+(?:\.\d+)?)\s*(year|month|week|day)", age_str, re.IGNORECASE)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("month"):
        value /= 12
    elif unit.startswith("week"):
        value /= 52
    elif unit.startswith("day"):
        value /= 365
    return value


def _extract_field(study: dict[str, Any], *path: str, default: Any = None) -> Any:
    node = study
    for key in path:
        if not isinstance(node, dict):
            return default
        node = node.get(key, default)
        if node is None:
            return default
    return node


def normalize_study(study: dict[str, Any]) -> dict[str, Any]:
    """Normalize a ClinicalTrials.gov v2 study dict to a flat Trial-compatible dict."""
    proto = study.get("protocolSection", {})
    id_mod = proto.get("identificationModule", {})
    status_mod = proto.get("statusModule", {})
    desc_mod = proto.get("descriptionModule", {})
    cond_mod = proto.get("conditionsModule", {})
    arms_mod = proto.get("armsInterventionsModule", {})
    elig_mod = proto.get("eligibilityModule", {})
    sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
    contacts_mod = proto.get("contactsLocationsModule", {})
    design_mod = proto.get("designModule", {})

    nct_id = id_mod.get("nctId", "")
    status_raw = status_mod.get("overallStatus", "UNKNOWN").upper()
    status = _STATUS_MAP.get(status_raw, TrialStatus.UNKNOWN)

    conditions = cond_mod.get("conditions", [])
    interventions = [i.get("name", "") for i in arms_mod.get("interventions", [])]
    phases = design_mod.get("phases", [])
    phase = phases[0] if phases else ""

    eligibility_text = elig_mod.get("eligibilityCriteria", "")
    gender_raw = elig_mod.get("sex", "ALL").upper()
    gender_map = {"MALE": "MALE", "FEMALE": "FEMALE", "ALL": "ALL"}
    gender = gender_map.get(gender_raw, "ALL")

    min_age = _parse_age_string(elig_mod.get("minimumAge"))
    max_age = _parse_age_string(elig_mod.get("maximumAge"))

    locations = list({
        loc.get("facility", "")
        for loc in contacts_mod.get("locations", [])
        if loc.get("facility")
    })

    last_updated_raw = status_mod.get("lastUpdatePostDateStruct", {}).get("date", "")
    start_date_raw = status_mod.get("startDateStruct", {}).get("date", "")
    primary_comp_raw = status_mod.get("primaryCompletionDateStruct", {}).get("date", "")

    return {
        "nct_id": nct_id,
        "title": id_mod.get("briefTitle", ""),
        "brief_summary": desc_mod.get("briefSummary", ""),
        "conditions": conditions,
        "interventions": interventions,
        "phase": phase,
        "status": status.value,
        "eligibility_text": eligibility_text,
        "eligibility_criteria": None,
        "sponsor": sponsor_mod.get("leadSponsor", {}).get("name", ""),
        "locations": locations,
        "start_date": start_date_raw or None,
        "last_updated": last_updated_raw or None,
        "cached_at": time.time(),
        "embedding": None,
        "_gender_hint": gender,
        "_min_age_years": min_age,
        "_max_age_years": max_age,
    }


class TrialFetcher:
    def __init__(self, config: TrialsConfig) -> None:
        self._config = config

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=2, max=60),
        retry=retry_if_exception_type(
            (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError)
        ),
        before_sleep=lambda retry_state: _logger.warning(
            "fetcher_retry",
            attempt=retry_state.attempt_number,
            error=str(retry_state.outcome.exception()) if retry_state.outcome else None,
        ),
    )
    async def _get_page(
        self,
        client: httpx.AsyncClient,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        resp = await client.get(
            f"{self._config.base_url}/studies",
            params=params,
            timeout=self._config.request_timeout_seconds,
        )
        resp.raise_for_status()
        return resp.json()

    async def fetch_by_condition(
        self,
        condition: str,
        status_filter: list[str] | None = None,
        max_trials: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yield normalized study dicts for the given condition query."""
        if status_filter is None:
            status_filter = ["RECRUITING"]
        max_trials = max_trials or self._config.max_trials_per_sync
        fetched = 0

        params: dict[str, Any] = {
            "query.cond": condition,
            "filter.overallStatus": ",".join(status_filter),
            "pageSize": min(self._config.page_size, max_trials),
            "fields": _FIELDS,
            "format": "json",
        }

        async with httpx.AsyncClient() as client:
            while fetched < max_trials:
                data = await self._get_page(client, params)
                studies = data.get("studies", [])
                for study in studies:
                    if fetched >= max_trials:
                        return
                    yield normalize_study(study)
                    fetched += 1

                next_token = data.get("nextPageToken")
                if not next_token:
                    break
                params["pageToken"] = next_token
