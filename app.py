"""
ClinicalTrial Match — Health Universe deployment
Calls the live Render API.
"""

import json

import requests
import streamlit as st

API = "https://clinicaltrial-match.onrender.com"
HEADERS = {"Content-Type": "application/json", "X-CTM-UI": "1"}

DEMO_FHIR = {
    "resourceType": "Bundle",
    "entry": [
        {"resource": {"resourceType": "Patient", "id": "demo-001", "birthDate": "1962-05-14", "gender": "male"}},
        {"resource": {"resourceType": "Condition", "code": {"text": "Type 2 Diabetes Mellitus"}, "clinicalStatus": {"text": "active"}}},
        {"resource": {"resourceType": "Condition", "code": {"text": "Hypertension"}, "clinicalStatus": {"text": "active"}}},
        {"resource": {"resourceType": "MedicationStatement", "medicationCodeableConcept": {"text": "Metformin"}, "status": "active"}},
        {"resource": {"resourceType": "MedicationStatement", "medicationCodeableConcept": {"text": "Lisinopril"}, "status": "active"}},
        {"resource": {"resourceType": "Observation", "code": {"text": "HbA1c"}, "valueQuantity": {"value": 8.2, "unit": "%"}}},
    ],
}

DEMO_NOTE = (
    "62-year-old male with a 10-year history of type 2 diabetes mellitus (HbA1c 8.2%) "
    "and hypertension, currently on Metformin 1000mg BID and Lisinopril 10mg daily. "
    "BMI 29. No significant cardiac history. Seeking optimisation of glycaemic control."
)

st.set_page_config(
    page_title="ClinicalTrial Match",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 ClinicalTrial Match")
st.caption("AI-powered matching of patients to recruiting clinical trials from ClinicalTrials.gov")

try:
    h = requests.get(f"{API}/v1/health", timeout=8).json()
    trials = h.get("trials_cached", 0)
    st.success(f"Service online · **{trials} trials** indexed")
except Exception:
    st.warning("⚠️ Service may be waking up. Refresh in 30s.")

st.divider()

tab1, tab2 = st.tabs(["🔬 Match Patient", "📋 Browse Trials"])

# ── Tab 1: Patient Matching ────────────────────────────────────────────────────
with tab1:
    st.subheader("Find Matching Clinical Trials")

    input_mode = st.radio("Input format", ["Clinical Note", "FHIR R4 Bundle"], horizontal=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        if input_mode == "Clinical Note":
            note = st.text_area("Clinical note", value=DEMO_NOTE, height=200)
        else:
            fhir_str = st.text_area(
                "FHIR R4 Bundle (JSON)",
                value=json.dumps(DEMO_FHIR, indent=2),
                height=300,
            )

    with col2:
        max_results = st.slider("Max results", 1, 20, 10)
        min_score = st.slider("Min match score", 0.0, 1.0, 0.15, step=0.05)
        status_filter = st.multiselect(
            "Trial status",
            ["RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING"],
            default=["RECRUITING"],
        )

    if st.button("🔍 Find Matching Trials", type="primary", use_container_width=True):
        with st.spinner("Analysing patient record and searching trials…"):
            try:
                if input_mode == "Clinical Note":
                    payload = {
                        "source": "note",
                        "note_text": note,
                        "max_results": max_results,
                        "min_score": min_score,
                        "trial_status_filter": status_filter or ["RECRUITING"],
                    }
                else:
                    try:
                        fhir_data = json.loads(fhir_str)
                    except json.JSONDecodeError as e:
                        st.error(f"Invalid JSON: {e}")
                        st.stop()
                    payload = {
                        "source": "fhir",
                        "fhir_data": fhir_data,
                        "max_results": max_results,
                        "min_score": min_score,
                        "trial_status_filter": status_filter or ["RECRUITING"],
                    }

                resp = requests.post(
                    f"{API}/v1/match/live",
                    json=payload,
                    headers=HEADERS,
                    timeout=60,
                )

                if resp.status_code == 200:
                    data = resp.json()
                    matches = data.get("matches", [])

                    # Patient summary
                    with st.expander("👤 Extracted Patient Profile", expanded=True):
                        c1, c2, c3, c4 = st.columns(4)
                        c1.metric("Age", f"{data.get('age_years') or '—'}y")
                        c2.metric("Gender", data.get("gender", "—").title())
                        c3.metric("Diagnoses", data.get("diagnoses_count", 0))
                        c4.metric("Medications", data.get("medications_count", 0))
                        if data.get("clinical_summary"):
                            st.caption(data["clinical_summary"])

                    if not matches:
                        st.info("No matches found. Try lowering the min score or syncing more trials in Browse Trials.")
                    else:
                        st.success(f"✅ Found **{len(matches)} matching trial(s)**")
                        for m in matches:
                            score = m.get("composite_score", 0)
                            color = "🟢" if score > 0.6 else ("🟡" if score > 0.3 else "🔴")
                            with st.expander(f"{color} {m.get('trial_title','Unknown')} — Score: {score:.0%}"):
                                col_a, col_b = st.columns([2, 1])
                                with col_a:
                                    st.markdown(f"**NCT ID:** `{m.get('nct_id','')}`")
                                    exp = m.get("explanation", {})
                                    if exp.get("summary"):
                                        st.markdown(f"**Summary:** {exp['summary']}")
                                    reasons = exp.get("key_reasons", [])
                                    if reasons:
                                        st.markdown("**Key reasons:**")
                                        for r in reasons:
                                            st.markdown(f"- {r}")
                                with col_b:
                                    st.metric("Match Score", f"{score:.0%}")
                                    st.metric("Confidence", m.get("confidence", "—"))
                                    st.link_button(
                                        "View on ClinicalTrials.gov ↗",
                                        f"https://clinicaltrials.gov/study/{m.get('nct_id','')}",
                                    )

                elif resp.status_code == 503:
                    st.warning(resp.json().get("detail", "Service loading — try again shortly."))
                elif resp.status_code == 402:
                    st.error("Payment required for API access. Use the hosted UI at clinicaltrial-match.onrender.com/ui")
                else:
                    st.error(f"Error {resp.status_code}: {resp.text[:300]}")

            except requests.exceptions.Timeout:
                st.warning("Request timed out — service may be starting. Try again in 30s.")
            except Exception as ex:
                st.error(f"Error: {ex}")

# ── Tab 2: Browse Trials ───────────────────────────────────────────────────────
with tab2:
    st.subheader("Browse Indexed Trials")

    col1, col2 = st.columns([3, 1])
    with col1:
        condition_filter = st.text_input("Filter by condition", placeholder="e.g. diabetes, cancer…")
    with col2:
        status_browse = st.selectbox(
            "Status",
            ["", "RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING", "COMPLETED"],
        )

    if st.button("🔄 Load Trials") or condition_filter or status_browse:
        with st.spinner("Loading…"):
            try:
                params = {"limit": 50}
                if condition_filter:
                    params["condition"] = condition_filter
                if status_browse:
                    params["status"] = status_browse

                resp = requests.get(f"{API}/v1/trials", params=params, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    trials_list = data.get("trials", [])
                    total = data.get("total", 0)
                    st.info(f"Showing {len(trials_list)} of {total} trials")

                    for t in trials_list:
                        with st.expander(f"`{t['nct_id']}` — {t['title']}"):
                            cols = st.columns([2, 1, 1])
                            with cols[0]:
                                st.markdown(f"**Conditions:** {', '.join(t.get('conditions', []))}")
                                if t.get("brief_summary"):
                                    st.caption(t["brief_summary"][:300] + "…")
                            with cols[1]:
                                st.markdown(f"**Status:** {t.get('status','—')}")
                                st.markdown(f"**Phase:** {t.get('phase','—')}")
                            with cols[2]:
                                st.markdown(f"**Sponsor:** {t.get('sponsor','—')}")
                                st.link_button("View ↗", f"https://clinicaltrials.gov/study/{t['nct_id']}")
                else:
                    st.error(f"Error {resp.status_code}")
            except Exception as ex:
                st.error(f"Error: {ex}")

st.divider()
st.caption("Powered by [ClinicalTrial Match](https://clinicaltrial-match.onrender.com/docs) · x402/MPP payment protocol")
