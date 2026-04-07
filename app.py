"""
ClinicalTrial Match — Health Universe deployment
Embeds the live Render UI directly so the look and feel is identical.
"""

import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="ClinicalTrial Match",
    page_icon="🧬",
    layout="wide",
)

# Minimal top bar — the Render UI has its own full navbar/header
st.markdown(
    """
    <style>
        /* Hide Streamlit's default header/footer padding so iframe fills the page */
        .block-container { padding-top: 0.5rem !important; padding-bottom: 0 !important; }
        header[data-testid="stHeader"] { display: none !important; }
        footer { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

components.iframe(
    "https://clinicaltrial-match.onrender.com",
    height=900,
    scrolling=True,
)
