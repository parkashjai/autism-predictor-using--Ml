# =========================================================
# NEUROIQ · ASD SCREENING & PREDICTION SYSTEM
# Styled after ChurnIQ — full dark sidebar, white cards,
# gauge chart, SHAP analysis, AI insights, recommendations
# =========================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import plotly.graph_objects as go
import plotly.express as px
import matplotlib
matplotlib.use("Agg")
import warnings
warnings.filterwarnings("ignore")

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="ASD Screening System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# GLOBAL CSS  (mirrors ChurnIQ palette & card style)
# =========================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;700&display=swap');

:root{
    --primary:#7c3aed;
    --secondary:#6366f1;
    --bg:#f4f7ff;
    --card:#ffffff;
    --text:#0f172a;
    --muted:#64748b;
    --border:#e2e8f0;
}

html, body, [data-testid="stAppViewContainer"]{
    background:var(--bg);
    font-family:'DM Sans',sans-serif;
}

.block-container{
    padding-top:2rem;
    padding-bottom:2rem;
}

/* ── Sidebar ── */
[data-testid="stSidebar"]{
    background:#0f172a !important;
}
[data-testid="stSidebar"] *{
    color:#e2e8f0 !important;
}
[data-testid="stSidebar"] label{
    color:#cbd5e1 !important;
    font-size:13px !important;
}

/* ── Selectbox (sidebar) ── */
[data-baseweb="select"] > div{
    background:#1e293b !important;
    border:1px solid #334155 !important;
    border-radius:12px !important;
}
[data-baseweb="select"] span{
    color:white !important;
}

/* ── Number input (sidebar) ── */
[data-testid="stNumberInput"] input{
    background:#1e293b !important;
    border:1px solid #334155 !important;
    border-radius:12px !important;
    color:#e2e8f0 !important;
}

/* ── Metrics ── */
div[data-testid="stMetric"]{
    background:white;
    border:1px solid #e2e8f0;
    border-radius:20px;
    padding:18px;
    box-shadow:0 4px 16px rgba(0,0,0,.04);
}
div[data-testid="stMetricValue"]{
    font-size:28px !important;
    font-weight:800 !important;
    color:#0f172a !important;
    font-family:'Syne',sans-serif !important;
}
div[data-testid="stMetricLabel"]{
    color:#64748b !important;
    font-size:12px !important;
}

/* ── Alerts ── */
[data-testid="stAlert"]{
    border-radius:18px !important;
}

/* ── Headings ── */
h1,h2,h3,h4{
    font-family:'Syne',sans-serif !important;
    color:#0f172a !important;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    try:
        return joblib.load("best_autism_pipeline.pkl")
    except FileNotFoundError:
        return None

model = load_model()

# =========================================================
# SESSION STATE
# =========================================================
if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

# =========================================================
# HEADER  (same card style as ChurnIQ)
# =========================================================
st.markdown("""
<div style="
background:white;
padding:28px;
border-radius:24px;
border:1px solid #e2e8f0;
display:flex;
justify-content:space-between;
align-items:center;
margin-bottom:24px;
box-shadow:0 4px 24px rgba(0,0,0,.04);
">

<div style="display:flex;align-items:center;gap:16px;">

<div style="
width:62px;
height:62px;
border-radius:18px;
background:linear-gradient(135deg,#7c3aed,#6366f1);
display:flex;
align-items:center;
justify-content:center;
font-size:28px;
box-shadow:0 8px 24px rgba(124,58,237,.25);
">
🧠
</div>

<div>
<div style="
font-size:34px;
font-weight:800;
font-family:Syne,sans-serif;
color:#0f172a;
line-height:1;
">
Autism Prediction
</div>
<div style="
font-size:15px;
color:#64748b;
margin-top:6px;
">
A real-time ASD risk prediction and clinical decision support tool
</div>
</div>
</div>

<div style="
background:#ecfdf5;
color:#166534;
border:1px solid #bbf7d0;
padding:10px 16px;
border-radius:12px;
font-size:13px;
font-weight:700;
">
🟢 Live Prediction
</div>

</div>
""", unsafe_allow_html=True)

if model is None:
    st.error("⚠️ Model file `best_autism_pipeline.pkl` not found. Place it in the same directory as this script.")
    st.stop()

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    st.title("🧬 Patient Profile")
    st.caption("Adjust patient attributes to screen")
    st.divider()

    
    st.subheader("📋 AQ-10 Behavioral Screening")
    st.caption("Select Yes if the statement describes the person.")

    with st.container(border=True):
        st.markdown("### 🎧 A1 — Sensory Awareness")
        st.write("Often notices small sounds that other people do not notice")
        a1 = st.radio("", ["No", "Yes"], horizontal=True,
                    label_visibility="collapsed", key="a1")
        a1 = 1 if a1 == "Yes" else 0

    with st.container(border=True):
        st.markdown("### 🔍 A2 — Attention to Detail")
        st.write("Prefers focusing on small details rather than the overall picture")
        a2 = st.radio("", ["No", "Yes"], horizontal=True,
                    label_visibility="collapsed", key="a2")
        a2 = 1 if a2 == "Yes" else 0

    with st.container(border=True):
        st.markdown("### 🔄 A3 — Multitasking")
        st.write("Finds it difficult to do multiple tasks at the same time")
        a3 = st.radio("", ["No", "Yes"], horizontal=True,
                    label_visibility="collapsed", key="a3")
        a3 = 1 if a3 == "Yes" else 0

    with st.container(border=True):
        st.markdown("### 📌 A4 — Flexibility")
        st.write("Finds it difficult to switch from one activity to another")
        a4 = st.radio("", ["No", "Yes"], horizontal=True,
                    label_visibility="collapsed", key="a4")
        a4 = 1 if a4 == "Yes" else 0

    with st.container(border=True):
        st.markdown("### 💬 A5 — Communication")
        st.write("Finds it difficult to understand implied meanings or indirect communication")
        a5 = st.radio("", ["No", "Yes"], horizontal=True,
                    label_visibility="collapsed", key="a5")
        a5 = 1 if a5 == "Yes" else 0

    with st.container(border=True):
        st.markdown("### 🔢 A6 — Pattern Recognition")
        st.write("Shows a strong interest in patterns, numbers, or routines")
        a6 = st.radio("", ["No", "Yes"], horizontal=True,
                    label_visibility="collapsed", key="a6")
        a6 = 1 if a6 == "Yes" else 0

    with st.container(border=True):
        st.markdown("### 🙂 A7 — Social Cues")
        st.write("Finds it difficult to understand facial expressions and body language")
        a7 = st.radio("", ["No", "Yes"], horizontal=True,
                    label_visibility="collapsed", key="a7")
        a7 = 1 if a7 == "Yes" else 0

    with st.container(border=True):
        st.markdown("### 😂 A8 — Figurative Language")
        st.write("Finds it difficult to understand jokes, sarcasm, or metaphors")
        a8 = st.radio("", ["No", "Yes"], horizontal=True,
                    label_visibility="collapsed", key="a8")
        a8 = 1 if a8 == "Yes" else 0

    with st.container(border=True):
        st.markdown("### 🎨 A9 — Imagination")
        st.write("Shows unusual or different imaginative interests")
        a9 = st.radio("", ["No", "Yes"], horizontal=True,
                    label_visibility="collapsed", key="a9")
        a9 = 1 if a9 == "Yes" else 0

    with st.container(border=True):
        st.markdown("### ⭐ A10 — Special Interests")
        st.write("Has a strong fascination with specific topics, dates, numbers, or patterns")
        a10 = st.radio("", ["No", "Yes"], horizontal=True,
                    label_visibility="collapsed", key="a10")
        a10 = 1 if a10 == "Yes" else 0
    # ── Demographics ──
    st.subheader("🏥 Demographics")

    age             = st.number_input("Age (years)", min_value=1, max_value=100, value=25)
    gender_display = st.radio(
    "Gender",
    ["Male", "Female"],
    horizontal=True
    )
    gender = "m" if gender_display == "Male" else "f"

    jaundice_display = st.radio(
        "Born with Jaundice?",
        ["No", "Yes"],
        horizontal=True
    )
    jaundice = "yes" if jaundice_display == "Yes" else "no"

    austim_display = st.radio(
        "Family History of Autism?",
        ["No", "Yes"],
        horizontal=True
    )
    austim = "yes" if austim_display == "Yes" else "no"

    used_app_before_display = st.radio(
        "Used Screening App Before?",
        ["No", "Yes"],
        horizontal=True
    )
    used_app_before = "yes" if used_app_before_display == "Yes" else "no"

    st.divider()

    # ── Background ──
    st.subheader("🌍 Background")

    ethnicity = st.selectbox("Ethnicity", [
        "White-European", "Latino", "Others", "Black", "Asian",
        "Middle Eastern", "South Asian", "Pasifika", "Hispanic", "Turkish"
    ])
    contry_of_res = st.selectbox("Country of Residence", [
        "United States", "India", "United Kingdom", "New Zealand",
        "Other_Country", "Canada", "United Arab Emirates", "Australia"
    ])
    relation = st.selectbox("Who is completing this test?", ["Self", "Parent", "Others"])

# =========================================================
# DETECT INTERACTION  (same pattern as ChurnIQ)
# =========================================================
default_vals = {"a1": 0, "a2": 0, "a3": 0, "a4": 0, "a5": 0,
                "a6": 0, "a7": 0, "a8": 0, "a9": 0, "a10": 0}

if any([a1, a2, a3, a4, a5, a6, a7, a8, a9, a10]):
    st.session_state.analysis_started = True

# =========================================================
# EMPTY STATE
# =========================================================
if not st.session_state.analysis_started:

    st.markdown("""
    <div style="
    background:white;
    padding:80px 40px;
    border-radius:30px;
    border:2px dashed #dbeafe;
    text-align:center;
    box-shadow:0 8px 30px rgba(0,0,0,.04);
    ">

    <div style="font-size:72px;margin-bottom:20px;">
    🧠
    </div>

    <div style="
    font-size:34px;
    font-weight:800;
    font-family:Syne,sans-serif;
    color:#0f172a;
    margin-bottom:14px;
    ">
    Live ASD Screening Analysis
    </div>

    <div style="
    font-size:16px;
    color:#64748b;
    max-width:700px;
    margin:auto;
    line-height:1.9;
    ">
    Configure patient behavioral scores and demographics in the sidebar
    to generate real-time ASD likelihood predictions, SHAP explainability,
    intelligent clinical insights, and recommended next steps.
    </div>

    </div>
    """, unsafe_allow_html=True)

# =========================================================
# MAIN ANALYSIS
# =========================================================
else:

    # ── Build input dataframe ──
    input_data = pd.DataFrame([{
        "A1_Score": a1,  "A2_Score": a2,  "A3_Score": a3,
        "A4_Score": a4,  "A5_Score": a5,  "A6_Score": a6,
        "A7_Score": a7,  "A8_Score": a8,  "A9_Score": a9,
        "A10_Score": a10,
        "age": age,
        "gender": gender,
        "ethnicity": ethnicity,
        "jaundice": jaundice,
        "austim": austim,
        "contry_of_res": contry_of_res,
        "used_app_before": used_app_before,
        "relation": relation,
    }])

    # ── Feature Engineering ──
    a_cols = [f"A{i}_Score" for i in range(1, 11)]

    # Total Behavioral Score
    input_data['Total_A_Score'] = input_data[a_cols].sum(axis=1)

    # Social Communication
    input_data['Social_Communication'] = (
        input_data['A1_Score']
        + input_data['A2_Score']
        + input_data['A3_Score']
        + input_data['A7_Score']
        + input_data['A8_Score']
    )

    # Repetitive Behaviors
    input_data['Repetitive_Behaviors'] = (
        input_data['A4_Score']
        + input_data['A5_Score']
        + input_data['A6_Score']
        + input_data['A9_Score']
        + input_data['A10_Score']
    )

    # Autism + Jaundice Risk
    input_data['Autism_Jaundice_Risk'] = (
        (input_data['austim'] == 'yes')
        & (input_data['jaundice'] == 'yes')
    ).astype(int)

    # Age Score Ratio
    input_data['Age_Score_Ratio'] = (
        input_data['Total_A_Score']
        / (input_data['age'] + 1)
    )

    # Values for display
    total_score = int(input_data['Total_A_Score'].iloc[0])
    social_score = int(input_data['Social_Communication'].iloc[0])
    rep_score = int(input_data['Repetitive_Behaviors'].iloc[0])
    # ── Predict ──
    try:
        prob = model.predict_proba(input_data)[0][1]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.stop()

    pct = round(prob * 100, 1)

    # Risk classification (mirrors ChurnIQ thresholds)
    if prob >= 0.66:
        risk_label = "High Risk"
        risk_color = "#ef4444"
    elif prob >= 0.31:
        risk_label = "Medium Risk"
        risk_color = "#f59e0b"
    else:
        risk_label = "Low Risk"
        risk_color = "#10b981"

    # =====================================================
    # PATIENT SNAPSHOT  (matches ChurnIQ "Customer Snapshot")
    # =====================================================
    st.subheader("🧬 Patient Snapshot")

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Age",            f"{age} yrs")
    s2.metric("AQ-10 Total",    f"{total_score} / 10")
    s3.metric("Social Score",   f"{social_score} / 5")
    s4.metric("Repetitive",     f"{rep_score} / 5")
    s5.metric("Family History", "Yes" if austim == "yes" else "No")

    st.markdown("<br>", unsafe_allow_html=True)

    # =====================================================
    # PREDICTION RESULTS
    # =====================================================
    st.subheader("📊 Prediction Results")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ASD Probability",     f"{pct}%")
    m2.metric("Non-ASD Probability", f"{round(100 - pct, 1)}%")
    m3.metric("Screening Threshold", "31%")
    m4.metric("Risk Level",          risk_label)

    # =====================================================
    # GAUGE CHART  (same Plotly Indicator as ChurnIQ)
    # =====================================================
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 40, "family": "Syne, sans-serif"}},
        title={"text": "ASD Likelihood Score", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8"},
            "bar":  {"color": risk_color, "thickness": 0.22},
            "steps": [
                {"range": [0,  30], "color": "#dcfce7"},
                {"range": [30, 65], "color": "#fef3c7"},
                {"range": [65, 100], "color": "#fee2e2"},
            ],
            "threshold": {
                "line":      {"color": risk_color, "width": 3},
                "thickness": 0.8,
                "value":     pct,
            },
        }
    ))
    fig.update_layout(height=320, margin=dict(t=60, b=0, l=30, r=30))
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # =====================================================
    # MODEL INSIGHT  (same pattern as ChurnIQ)
    # =====================================================
    st.subheader("🧠 Model Insight")

    if prob >= 0.66:
        st.error("🚨 Strong ASD indicators detected. A formal diagnostic assessment is strongly recommended.")
    elif prob >= 0.31:
        st.warning("⚠️ Moderate ASD risk detected. Some behavioral patterns suggest further clinical evaluation.")
    else:
        st.success("✅ No significant ASD indicators found. Routine monitoring is sufficient.")

    # =====================================================
    # SHAP FEATURE ANALYSIS  (Plotly bar — matches ChurnIQ)
    # =====================================================
    st.subheader("📈 SHAP Feature Analysis")

    try:
        preprocessor = model.named_steps["preprocessor"]
        clf          = model.named_steps["model"]
        X_transformed = preprocessor.transform(input_data)

        # Reconstruct feature names
        binary_cols  = ["gender", "jaundice", "austim", "used_app_before"]
        nominal_cols = ["ethnicity", "contry_of_res", "relation"]
        ohe          = preprocessor.named_transformers_["nom_encode"]
        ohe_names    = list(ohe.get_feature_names_out(nominal_cols))
        remainder_cols = [c for c in input_data.columns
                          if c not in binary_cols + nominal_cols]
        all_feature_names = binary_cols + ohe_names + remainder_cols

        if X_transformed.shape[1] != len(all_feature_names):
            all_feature_names = [f"feature_{i}" for i in range(X_transformed.shape[1])]

        X_df = pd.DataFrame(X_transformed, columns=all_feature_names)

        explainer   = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_df)

        sv = np.array(shap_values)

        if sv.ndim == 3:
            sv = sv[0, :, 1]
        elif sv.ndim == 2:
            sv = sv[0]

        sv = np.array(sv).flatten()

        shap_df = pd.DataFrame({
            "Feature": all_feature_names,
            "SHAP":    sv
        })
        shap_df["Impact"] = shap_df["SHAP"].abs()
        shap_df = shap_df.sort_values("Impact", ascending=False).head(10)

        # Clean feature names (strip encoder prefixes)
        shap_df["Feature"] = shap_df["Feature"].str.replace(
            r"^(bin_encode__|nom_encode__|remainder__)", "", regex=True
        )

        fig2 = px.bar(
            shap_df,
            x="SHAP",
            y="Feature",
            orientation="h",
            text_auto=".3f",
            color="SHAP",
            color_continuous_scale="RdYlGn_r",
            labels={"SHAP": "SHAP Value (impact on prediction)", "Feature": ""},
        )
        fig2.update_layout(
            height=420,
            plot_bgcolor="white",
            paper_bgcolor="white",
            coloraxis_showscale=False,
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis={"autorange": "reversed"},
        )
        fig2.update_traces(textfont_size=11)
        st.plotly_chart(fig2, use_container_width=True)

    except Exception as e:
        st.error(f"SHAP unavailable: {e}")

    # =====================================================
    # AI CLINICAL ANALYSIS  (mirrors "AI Customer Analysis")
    # =====================================================
    st.subheader("🧠 AI Clinical Analysis")

    analysis_points = []

    # AQ-10 Total Score
    if total_score >= 7:
        analysis_points.append(("⚠️", "High Risk",
            f"AQ-10 total score is {total_score}/10. Scores ≥ 7 are considered a strong indicator for ASD assessment referral."))
    elif total_score >= 4:
        analysis_points.append(("🟡", "Moderate",
            f"AQ-10 total score is {total_score}/10, suggesting some behavioral traits worth monitoring with a specialist."))
    else:
        analysis_points.append(("✅", "Positive",
            f"AQ-10 total score is {total_score}/10 — below the clinical screening threshold."))

    # Social Communication
    if social_score >= 4:
        analysis_points.append(("⚠️", "High Risk",
            f"Social communication score is {social_score}/5 — significant difficulty with social interaction and communication detected."))
    elif social_score >= 2:
        analysis_points.append(("🟡", "Moderate",
            f"Social communication score is {social_score}/5 — some challenges noted in social interaction."))
    else:
        analysis_points.append(("✅", "Positive",
            f"Social communication score is {social_score}/5 — no major social interaction concerns."))

    # Repetitive Behaviors
    if rep_score >= 4:
        analysis_points.append(("⚠️", "High Risk",
            f"Repetitive behavior score is {rep_score}/5 — high frequency of restricted and repetitive behavior patterns."))
    elif rep_score >= 2:
        analysis_points.append(("🟡", "Moderate",
            f"Repetitive behavior score is {rep_score}/5 — some repetitive tendencies observed."))
    else:
        analysis_points.append(("✅", "Positive",
            f"Repetitive behavior score is {rep_score}/5 — minimal repetitive patterns detected."))

    # Family History
    if austim == "yes":
        analysis_points.append(("⚠️", "High Risk",
            "Family history of autism is present. Genetic predisposition significantly increases risk and warrants clinical evaluation."))
    else:
        analysis_points.append(("✅", "Positive",
            "No family history of autism reported, reducing hereditary risk factor contribution."))

    # Jaundice at birth
    if jaundice == "yes":
        analysis_points.append(("🟡", "Moderate",
            "Neonatal jaundice history noted. Research links early jaundice to slightly elevated neurodevelopmental risk."))
    else:
        analysis_points.append(("✅", "Positive",
            "No neonatal jaundice reported — no associated perinatal risk factor detected."))

    # Who is completing (Self = stronger signal)
    if relation == "Self":
        analysis_points.append(("🟡", "Moderate",
            "Test completed by the individual themselves. Self-reported assessments can carry higher accuracy but also self-referral bias."))
    elif relation == "Parent":
        analysis_points.append(("✅", "Positive",
            "Test completed by a parent, providing a caregiver perspective that often adds reliability to the screening."))
    else:
        analysis_points.append(("✅", "Positive",
            "Test completed by a third party. Multi-perspective screening improves diagnostic confidence."))

    # Render analysis cards (identical card style to ChurnIQ)
    for icon, severity, text in analysis_points:

        if severity == "High Risk":
            bg     = "#fef2f2"
            border = "#fecaca"
            color  = "#991b1b"
        elif severity == "Moderate":
            bg     = "#fffbeb"
            border = "#fde68a"
            color  = "#92400e"
        else:
            bg     = "#f0fdf4"
            border = "#bbf7d0"
            color  = "#166534"

        st.markdown(f"""
        <div style="
        background:{bg};
        border:1px solid {border};
        border-radius:18px;
        padding:18px;
        margin-bottom:14px;
        ">
        <div style="font-size:15px;font-weight:700;color:{color};margin-bottom:6px;">
            {icon} {severity}
        </div>
        <div style="color:#334155;font-size:14px;line-height:1.8;">
            {text}
        </div>
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # PREDICTION SUMMARY  (matches "Business Summary")
    # =====================================================
    st.subheader("📘 Screening Summary")

    if prob >= 0.66:
        summary = """
        This individual demonstrates multiple strong ASD behavioral indicators
        across both social communication and repetitive behavior domains.
        A comprehensive clinical evaluation by a certified specialist is strongly recommended
        at the earliest opportunity.
        """
    elif prob >= 0.31:
        summary = """
        Some behavioral patterns consistent with ASD traits have been identified.
        While this is not a diagnosis, the screening suggests that further discussion
        with a developmental pediatrician or psychologist would be beneficial.
        """
    else:
        summary = """
        The screening did not identify significant ASD indicators at this time.
        Behavioral scores are within expected ranges. If concerns persist,
        routine follow-up with a healthcare provider is always appropriate.
        """

    st.markdown(f"""
    <div style="
    background:#f8fafc;
    border:1px solid #e2e8f0;
    border-radius:16px;
    padding:18px 20px;
    margin-top:10px;
    ">
    <div style="font-size:14px;line-height:1.9;color:#475569;font-weight:500;">
        {summary}
    </div>
    </div>
    """, unsafe_allow_html=True)

    # =====================================================
    # RECOMMENDED ACTIONS  (matches ChurnIQ recommendations)
    # =====================================================
    st.subheader("🎯 Recommended Actions")

    if prob >= 0.66:
        recommendations = [
            "🏥 Refer to a developmental pediatrician or psychiatrist immediately",
            "📋 Schedule a comprehensive ADOS-2 / ADI-R diagnostic evaluation",
            "👨‍👩‍👧 Engage family in psychoeducation and early support planning",
            "📞 Connect with local ASD support and intervention services",
        ]
    elif prob >= 0.31:
        recommendations = [
            "🩺 Schedule a follow-up consultation with a specialist",
            "📊 Monitor behavioral development over the next 3–6 months",
            "📚 Provide family with ASD awareness and early intervention resources",
        ]
    else:
        recommendations = [
            "✅ Continue routine developmental monitoring",
            "📅 Schedule next standard developmental screening at appropriate age",
            "💬 Reassure family — no significant ASD indicators found at this time",
        ]

    for rec in recommendations:
        st.markdown(f"""
        <div style="
        background:white;
        border:1px solid #e2e8f0;
        border-radius:16px;
        padding:16px;
        margin-bottom:12px;
        box-shadow:0 2px 10px rgba(0,0,0,.03);
        font-weight:500;
        color:#0f172a;
        font-size:14px;
        ">
        {rec}
        </div>
        """, unsafe_allow_html=True)

    # =====================================================
    # RAW INPUT EXPANDER
    # =====================================================
    with st.expander("🗂️ View Raw Input Data"):
        st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)

    # =====================================================
    # FOOTER  (matches ChurnIQ footer)
    # =====================================================
    st.markdown("""
    <div style="
    margin-top:30px;
    background:white;
    padding:18px;
    border-radius:18px;
    border:1px solid #e2e8f0;
    text-align:center;
    color:#64748b;
    font-size:13px;
    ">
    NeuroIQ · ASD Screening & Clinical Decision Support System ·
    <span style="color:#ef4444;font-weight:600;">Not a clinical diagnosis</span> —
    always consult a qualified healthcare professional.
    </div>
    """, unsafe_allow_html=True)
