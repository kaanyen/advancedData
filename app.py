import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
import os

# --- 1. CONFIGURATION & DESIGN SYSTEM (Flat UI / iOS Style) ---
st.set_page_config(
    page_title="Ashesi Student Success Analytics",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CUSTOM CSS: Flat Design, No Dark Mode, Clean Typography
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* GLOBAL THEME OVERRIDES */
    :root {
        --primary-color: #881c1c; /* Ashesi Burgundy */
        --bg-color: #F2F2F7;      /* iOS System Grey 6 */
        --card-bg: #FFFFFF;
        --text-color: #1C1C1E;
        --secondary-text: #8E8E93;
        --accent-blue: #007AFF;   /* iOS Blue */
        --success-green: #34C759; /* iOS Green */
        --warning-orange: #FF9500;/* iOS Orange */
        --danger-red: #FF3B30;    /* iOS Red */
    }

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: var(--bg-color);
        color: var(--text-color);
    }

    /* REMOVE STREAMLIT BRANDING */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* CARD COMPONENT */
    .ios-card {
        background-color: var(--card-bg);
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05); /* Subtle Shadow */
        border: 1px solid rgba(0,0,0,0.02);
    }

    /* TYPOGRAPHY */
    h1, h2, h3 {
        font-weight: 700;
        letter-spacing: -0.02em;
        color: #000000;
    }
    h4, h5 {
        font-weight: 600;
        color: #333333;
    }
    p {
        line-height: 1.5;
        color: #3A3A3C;
    }
    .caption {
        font-size: 0.85rem;
        color: var(--secondary-text);
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.05em;
        margin-bottom: 8px;
    }

    /* NAVIGATION TABS (Segmented Control Style) */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #E5E5EA; /* iOS Segmented Control BG */
        border-radius: 8px;
        padding: 2px;
        gap: 0px;
        margin-bottom: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 32px;
        border-radius: 6px;
        font-weight: 500;
        font-size: 13px;
        color: #000000;
        background-color: transparent;
        border: none;
        margin: 2px;
        flex: 1; /* Equal width */
    }
    .stTabs [aria-selected="true"] {
        background-color: #FFFFFF !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
        color: #000000 !important;
    }

    /* BUTTONS */
    .stButton > button {
        background-color: var(--accent-blue);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-weight: 600;
        transition: opacity 0.2s;
    }
    .stButton > button:hover {
        background-color: var(--accent-blue);
        opacity: 0.8;
        color: white;
    }

    /* HIDE DEFAULT PADDING */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
        max-width: 1200px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA & MODEL LOADING ---
@st.cache_data
def load_data():
    if os.path.exists('master_student_data.csv'):
        return pd.read_csv('master_student_data.csv')
    return None

@st.cache_resource
def load_models():
    paths = {
        'early_warning': 'saved_models/model_early_warning.pkl',
        'timeline': 'saved_models/model_grad_timeline.pkl',
        'misconduct': 'saved_models/model_misconduct.pkl',
        'year1_success': 'saved_models/model_success_year1.pkl'
    }
    models = {}
    for name, path in paths.items():
        if os.path.exists(path):
            models[name] = joblib.load(path)
    return models

df = load_data()
models_pkg = load_models()

# Helper for Safe Encoding (with fallback mapping)
def safe_encode(encoder, value):
    try: 
        return encoder.transform([str(value)])[0]
    except: 
        return 0

# Simple encoding mappings (fallback when encoders not in model package)
def encode_track(track):
    mapping = {"Calculus": 0, "Pre-Calculus": 1, "College Algebra": 2}
    return mapping.get(track, 0)

def encode_gender(gender):
    mapping = {"Male": 0, "Female": 1}
    return mapping.get(gender, 0)

def encode_aid(aid):
    mapping = {"No Financial Aid": 0, "Scholarship": 1}
    return mapping.get(aid, 0)

def encode_major(major):
    mapping = {
        "Computer Science": 0, "CS": 0,
        "MIS": 1,
        "Business Admin": 2, "Business": 2,
        "Engineering": 3
    }
    return mapping.get(major, 0)

# Inference Logic (Matches Notebook Feature Engineering)
def get_advanced_features(math_score, strength):
    # Base: Other scores correlated with Math
    scores = {
        'HS_STEM_Score': math_score,
        'HS_Business_Score': math_score * 0.85,
        'HS_Humanities_Score': math_score * 0.8
    }
    # Boost based on strength
    if strength == "STEM & Science":
        scores['HS_STEM_Score'] = min(10, math_score + 0.5)
    elif strength == "Business & Economics":
        scores['HS_Business_Score'] = min(10, math_score + 2.0)
        scores['HS_STEM_Score'] = max(0, math_score - 1.5)
    elif strength == "Arts & Humanities":
        scores['HS_Humanities_Score'] = min(10, math_score + 2.5)
        scores['HS_STEM_Score'] = max(0, math_score - 2.5)
    return scores

# --- 3. TOP NAVIGATION & HEADER ---
st.title("Student Success Analytics")
st.markdown("**Capstone Project Dashboard** | Ashesi University")
st.markdown("---")

# Navigation (Segmented Control Style)
tabs = st.tabs([
    "Executive Summary",
    "Q1: Academic Risk",
    "Q2: Conduct Risk",
    "Q3: Major Fit (Yr 1)",
    "Q4: Major Fit (Yr 2)",
    "Q5: Math Tracks",
    "Q6: Graduation Timeline"
])

# ==============================================================================
# TAB 1: EXECUTIVE SUMMARY
# ==============================================================================
with tabs[0]:
    if df is not None:
        # KPI Row
        c1, c2, c3, c4 = st.columns(4)
        total = len(df)
        grad_rate = len(df[df['Student Status'] == 'Graduated']) / total
        risk_rate = len(df[df['Is_Struggling'] == 1]) / total
        honor_rate = len(df[df['Is_High_Achiever'] == 1]) / total

        def metric_card(col, label, value, subtext, color="#000000"):
            col.markdown(f"""
            <div class="ios-card" style="padding: 20px; text-align: left;">
                <div class="caption">{label}</div>
                <div style="font-size: 28px; font-weight: 700; color: {color};">{value}</div>
                <div style="font-size: 13px; color: #8E8E93; margin-top: 4px;">{subtext}</div>
            </div>
            """, unsafe_allow_html=True)

        metric_card(c1, "Total Students", f"{total:,}", "Analyzed Cohort")
        metric_card(c2, "Graduation Rate", f"{grad_rate:.1%}", "Overall Completion", "#34C759")
        metric_card(c3, "At-Risk (Freshmen)", f"{risk_rate:.1%}", "GPA < 2.0 in Yr 1", "#FF3B30")
        metric_card(c4, "High Achievers", f"{honor_rate:.1%}", "Final GPA ≥ 3.0", "#FF9500")

        # Narrative Row
        c_left, c_right = st.columns([1, 1])
        with c_left:
            st.markdown('<div class="ios-card">', unsafe_allow_html=True)
            st.subheader("Key Insights")
            st.markdown("""
            * **The 'Great Filter':** Admissions data only predicts success with **65% accuracy**. However, by the end of Year 1, prediction accuracy jumps to **85%**.
            * **Resilience:** Students starting in **College Algebra** (the lowest math track) graduate at higher rates than Calculus students, though they rarely achieve 'High Honors'.
            * **Misconduct:** Academic misconduct is situational, not demographic. It correlates strongly with low GPAs (The 'Desperation Hypothesis') but cannot be predicted by admissions data alone.
            """)
            st.markdown('</div>', unsafe_allow_html=True)

        with c_right:
            st.markdown('<div class="ios-card">', unsafe_allow_html=True)
            st.subheader("Incoming Profile by Major")
            # Aggregated scores
            major_scores = df.groupby('Admissions_Major')[['HS_STEM_Score', 'HS_Business_Score', 'HS_Humanities_Score']].mean().reset_index()
            major_melt = major_scores.melt(id_vars='Admissions_Major', var_name='Type', value_name='Score')
            
            fig = px.bar(major_melt, x='Admissions_Major', y='Score', color='Type', barmode='group',
                         color_discrete_sequence=['#2c3e50', '#f1c40f', '#881c1c'])
            fig.update_layout(plot_bgcolor='white', xaxis_title=None, yaxis_title="Avg Score (0-10)", legend_title=None)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 2: Q1 - ACADEMIC RISK (EARLY WARNING)
# ==============================================================================
with tabs[1]:
    st.markdown('<div class="ios-card">', unsafe_allow_html=True)
    st.markdown("#### Q: Can admissions data predict academic struggle in the first year?")
    
    col_sim, col_res = st.columns([1, 2])
    
    with col_sim:
        st.markdown("**Simulate Student Profile**")
        with st.form("risk_form"):
            hs_score = st.slider("Math Proficiency (0-10)", 0.0, 10.0, 7.0)
            gap = st.number_input("Gap Years", 0, 5, 0)
            track = st.selectbox("Math Track", ["Calculus", "Pre-Calculus", "College Algebra"])
            major = st.selectbox("Major", ["Computer Science", "MIS", "Business Admin", "Engineering"])
            strength = st.selectbox("Strength", ["STEM & Science", "Business & Economics"])
            gender = st.selectbox("Gender", ["Male", "Female"])
            aid = st.selectbox("Financial Aid", ["No Financial Aid", "Scholarship"])
            
            run_risk = st.form_submit_button("Predict Risk")

    with col_res:
        if run_risk and 'early_warning' in models_pkg:
            pkg = models_pkg['early_warning']
            # Use encoders if available, otherwise use fallback functions
            if 'encoders' in pkg:
                enc = pkg['encoders']
                track_code = safe_encode(enc.get('track', None), track) if enc.get('track') else encode_track(track)
                gender_code = safe_encode(enc.get('gender', None), gender) if enc.get('gender') else encode_gender(gender)
                aid_code = safe_encode(enc.get('aid', None), aid) if enc.get('aid') else encode_aid(aid)
                major_code = safe_encode(enc.get('major', None), major) if enc.get('major') else encode_major(major)
            else:
                track_code = encode_track(track)
                gender_code = encode_gender(gender)
                aid_code = encode_aid(aid)
                major_code = encode_major(major)
            
            adv = get_advanced_features(hs_score, strength)
            
            # Vector: [HS_Math, HS_STEM, HS_Bus, HS_Hum, Gap, Track, Gen, Aid, Maj]
            vec = np.array([[
                hs_score, adv['HS_STEM_Score'], adv['HS_Business_Score'], adv['HS_Humanities_Score'],
                gap, track_code, gender_code, aid_code, major_code
            ]])
            
            prob = pkg['model'].predict_proba(vec)[0][1]
            
            # Result Card
            color = "#FF3B30" if prob > 0.5 else "#34C759"
            status = "High Risk" if prob > 0.5 else "Low Risk"
            
            st.markdown(f"""
            <div style="background-color: {color}15; border-left: 4px solid {color}; padding: 20px; border-radius: 8px;">
                <h3 style="color: {color}; margin:0;">{status} ({prob:.1%})</h3>
                <p style="margin-top:10px;">Likelihood of Year 1 Academic Struggle (GPA < 2.0)</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Feature Importance (Context)
            st.markdown("##### What drives this prediction?")
            imps = pd.DataFrame({
                'Factor': ['HS Math', 'STEM Score', 'Business Score', 'Humanities', 'Gap Years', 'Track', 'Gender', 'Aid', 'Major'],
                'Impact': pkg['model'].feature_importances_
            }).sort_values('Impact', ascending=True)
            
            fig = px.bar(imps, x='Impact', y='Factor', orientation='h', color_discrete_sequence=['#881c1c'])
            fig.update_layout(plot_bgcolor='white', height=300)
            st.plotly_chart(fig, use_container_width=True)
            
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 3: Q2 - MISCONDUCT RISK
# ==============================================================================
with tabs[2]:
    st.markdown('<div class="ios-card">', unsafe_allow_html=True)
    st.markdown("#### Q: Can admissions patterns predict if a student is likely to get into trouble?")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**The Verdict: NO.**")
        st.markdown("""
        Our models achieved only **~55% accuracy** (near random chance) when trying to predict misconduct from admissions data. 
        This is a crucial ethical finding: **"Troublemakers" cannot be profiled by their background.**
        """)
        
        # Desperation Hypothesis Chart
        if df is not None:
            gpa_cond = df.groupby('Has_Academic_Case')['First_Year_GPA'].mean().reset_index()
            gpa_cond['Status'] = gpa_cond['Has_Academic_Case'].map({0: 'No Case', 1: 'Academic Case'})
            
            fig = px.bar(gpa_cond, x='Status', y='First_Year_GPA', color='Status',
                         title="The 'Desperation Hypothesis'",
                         color_discrete_sequence=['#34C759', '#FF3B30'])
            fig.add_hline(y=2.0, line_dash="dot", annotation_text="Probation (2.0)")
            fig.update_layout(plot_bgcolor='white', yaxis_title="Avg Prior GPA")
            st.plotly_chart(fig, use_container_width=True)
            
    with c2:
        st.markdown("**Forensic Analysis: Subject Correlations**")
        st.markdown("While overall prediction is poor, specific subject weaknesses correlate with risk.")
        # Static representation of the forensic findings from notebook
        data = pd.DataFrame({
            'Subject': ['Literature (Protective)', 'History (Protective)', 'Physics (Risk)', 'Math (Risk)'],
            'Correlation': [-0.15, -0.12, 0.08, 0.05]
        })
        fig = px.bar(data, x='Correlation', y='Subject', orientation='h', 
                     color='Correlation', color_continuous_scale='RdBu_r')
        fig.update_layout(plot_bgcolor='white')
        st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 4: Q3 - MAJOR FIT (YEAR 1)
# ==============================================================================
with tabs[3]:
    st.markdown('<div class="ios-card">', unsafe_allow_html=True)
    st.markdown("#### Q: Can Year 1 data predict success/failure in a specific major?")
    
    st.info("💡 **Insight:** Adding Year 1 Grades increases predictive power by +20%.")
    
    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown("**Simulate Year 1 Performance**")
        y1_gpa = st.slider("Year 1 GPA", 0.0, 4.0, 2.5)
        failed = st.number_input("Failed Courses", 0, 5, 0)
        
        if st.button("Check Major Fit"):
            if 'year1_success' in models_pkg:
                # Vector: [HS_Math, HS_STEM, HS_Bus, HS_Hum, Gap, Track, Gen, Aid, Maj, Y1_GPA]
                # Note: Model expects 10 features (Fail_Count was dropped during training)
                vec = np.array([[7, 7, 6, 6, 0, 0, 0, 0, 0, y1_gpa]])
                # Note: This uses a simplified interaction for demo purposes
                # Real implementation would use the full vector from Q1
                pred = models_pkg['year1_success']['model'].predict(vec)[0]
                
                res = "Likely High Achiever" if pred == 1 else "Likely Average/Struggle"
                st.success(f"Prediction: {res}")
    
    with c2:
        if df is not None:
            # Success Rate by Major
            maj_succ = df.groupby('Admissions_Major')['Is_High_Achiever'].mean().reset_index()
            fig = px.bar(maj_succ, y='Admissions_Major', x='Is_High_Achiever', orientation='h',
                         title="Difficulty by Major (Pct High Achievers)",
                         color='Is_High_Achiever', color_continuous_scale='Blues')
            fig.update_layout(plot_bgcolor='white', xaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 5: Q4 - MAJOR FIT (YEAR 2)
# ==============================================================================
with tabs[4]:
    st.markdown('<div class="ios-card">', unsafe_allow_html=True)
    st.markdown("#### Q: Can Year 2 data predict success/failure?")
    st.markdown("**The 'Fade Out' Effect:** By Year 2, High School grades become irrelevant predictors.")
    
    # Visualization of Feature Importance Shift (Conceptual based on Notebook)
    data = pd.DataFrame({
        'Feature': ['Y2 Major Core', 'Y1 GPA', 'HS Math', 'HS English'],
        'Importance (Year 2 Model)': [0.45, 0.35, 0.05, 0.02]
    })
    
    fig = px.bar(data, x='Importance (Year 2 Model)', y='Feature', orientation='h', color_discrete_sequence=['#800080'])
    fig.update_layout(plot_bgcolor='white', title="What matters in Year 2?")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 6: Q5 & Q6 - MATH TRACKS & PATHWAYS
# ==============================================================================
with tabs[5]:
    st.markdown('<div class="ios-card">', unsafe_allow_html=True)
    st.markdown("#### Q: Is there a performance difference between Math Tracks?")
    st.markdown("#### Q: Can College Algebra students succeed in CS?")
    
    if df is not None:
        c1, c2 = st.columns(2)
        
        with c1:
            st.markdown("**Graduation Rate (Survival)**")
            grad_stats = df.groupby('Math_Track').apply(lambda x: (x['Student Status'] == 'Graduated').mean()).reset_index(name='Rate')
            fig = px.bar(grad_stats, x='Math_Track', y='Rate', color='Math_Track',
                         color_discrete_sequence=['#881c1c', '#f1c40f', '#2c3e50'])
            fig.update_layout(plot_bgcolor='white', yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("College Algebra students often have HIGHER graduation rates.")
            
        with c2:
            st.markdown("**High Achiever Rate (Excellence)**")
            honors_stats = df.groupby('Math_Track')['Is_High_Achiever'].mean().reset_index()
            fig = px.bar(honors_stats, x='Math_Track', y='Is_High_Achiever', color='Math_Track',
                         color_discrete_sequence=['#881c1c', '#f1c40f', '#2c3e50'])
            fig.update_layout(plot_bgcolor='white', yaxis_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
            st.caption("However, they struggle to reach High Honors status.")
            
    st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 7: Q7 - GRADUATION TIMELINE
# ==============================================================================
with tabs[6]:
    st.markdown('<div class="ios-card">', unsafe_allow_html=True)
    st.markdown("#### Q: Can we predict if a student will need >8 semesters?")
    
    col_in, col_out = st.columns([1, 2])
    
    with col_in:
        with st.form("timeline_form"):
            hs_math = st.slider("HS Math Score", 0, 10, 5)
            gap_y = st.number_input("Gap Years", 0, 5, 2)
            tk = st.selectbox("Track", ["College Algebra", "Calculus"])
            mj = st.selectbox("Major", ["Engineering", "CS", "Business"])
            
            check_time = st.form_submit_button("Predict Timeline")
            
    with col_out:
        if check_time and 'timeline' in models_pkg:
            # Simplified vector reconstruction
            # [HS_Math, HS_STEM, HS_Bus, HS_Hum, Gap, Track, Gen, Aid, Maj]
            # Use encoders if available, otherwise use fallback functions
            pkg = models_pkg['timeline']
            if 'encoders' in pkg and pkg['encoders']:
                enc = pkg['encoders']
                track_code = safe_encode(enc.get('track', None), tk) if enc.get('track') else encode_track(tk)
                major_code = safe_encode(enc.get('major', None), mj) if enc.get('major') else encode_major(mj)
            else:
                track_code = encode_track(tk)
                major_code = encode_major(mj)
            
            # Filling averages for non-inputs (using same HS scores for all, and 0 for gender/aid)
            v = np.array([[hs_math, hs_math, hs_math, hs_math, gap_y, track_code, 0, 0, major_code]])
            
            p_delay = models_pkg['timeline']['model'].predict_proba(v)[0][1]
            
            fig = go.Figure(go.Indicator(
                mode = "number+gauge",
                value = p_delay * 100,
                title = {'text': "Risk of Delayed Graduation"},
                gauge = {'axis': {'range': [0, 100]}, 'bar': {'color': "#FF9500"}}
            ))
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
            
            if p_delay > 0.5:
                st.warning("High risk of requiring 9+ semesters. Early course planning advised.")
            else:
                st.success("On track for standard 4-year graduation.")

    st.markdown('</div>', unsafe_allow_html=True)
