import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
import os

# --- 1. CONFIGURATION & MATERIAL 3 DESIGN SYSTEM ---
st.set_page_config(
    page_title="Student Success Command Center",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CUSTOM CSS: Professional Material 3 Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Roboto+Mono:wght@400;500&family=Roboto:wght@300;400;500&display=swap');

    /* BASE LAYOUT */
    .stApp {
        background-color: #f8f9fa; /* Light Grey Background */
    }
    
    h1, h2, h3, h4, h5 {
        font-family: 'Outfit', sans-serif;
        color: #2c3e50;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    p, div, span, label {
        font-family: 'Roboto', sans-serif;
        color: #49454f;
    }

    /* MATERIAL 3 CARD SYSTEM */
    .glass-card {
        background-color: white;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        border: 1px solid #e9ecef;
        margin-bottom: 1.5rem;
        transition: box-shadow 0.2s;
    }
    .glass-card:hover {
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }

    /* METRIC HIGHLIGHTS */
    .big-stat {
        font-family: 'Outfit', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #881c1c;
        line-height: 1;
    }
    .stat-label {
        font-family: 'Roboto Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        color: #6c757d;
        margin-top: 0.5rem;
    }

    /* CUSTOM TABS (PILL STYLE) */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        padding: 8px;
        border-radius: 50px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
        margin-bottom: 2rem;
        display: flex;
        justify-content: center;
        width: fit-content;
        margin-left: auto;
        margin-right: auto;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        border-radius: 20px;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        font-size: 14px;
        border: none;
        background-color: transparent;
        padding: 0 24px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2c3e50 !important;
        color: white !important;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }

    /* ALERT & INSIGHT BOXES */
    .insight-box {
        border-left: 4px solid #f1c40f;
        background: #fffdf5;
        padding: 1.25rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.95rem;
        line-height: 1.5;
    }
    
    /* BUTTON STYLING */
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        font-family: 'Outfit', sans-serif;
        font-weight: 600;
        padding: 0.5rem 1rem;
    }

    /* REMOVE DEFAULT PADDING */
    .block-container { padding-top: 2rem; max-width: 1200px; }
    </style>
    """, unsafe_allow_html=True)

# COLOR PALETTE
ASHESI_RED = '#881c1c'
ASHESI_GOLD = '#f1c40f'
ASHESI_SLATE = '#2c3e50'
ASHESI_GREY = '#95a5a6'

# --- 2. LOGIC ENGINE (DATA & MODELS) ---
@st.cache_data
def load_data():
    if os.path.exists('master_student_data.csv'):
        return pd.read_csv('master_student_data.csv')
    return None

@st.cache_resource
def load_models():
    # Looks for models in 'saved_models' folder
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

def safe_encode(encoder, value):
    try: return encoder.transform([str(value)])[0]
    except: return 0

# SMART FEATURE INFERENCE (The Bridge between User and 9-Feature Model)
def get_advanced_features(math_score, strength):
    # Base assumption: Other scores correlate with Math
    scores = {
        'HS_STEM_Score': math_score,
        'HS_Business_Score': math_score * 0.85,
        'HS_Humanities_Score': math_score * 0.8
    }
    # Adjust based on strength
    if strength == "STEM & Science":
        scores['HS_STEM_Score'] = min(10, math_score + 0.5)
    elif strength == "Business & Economics":
        scores['HS_Business_Score'] = min(10, math_score + 2.0)
        scores['HS_STEM_Score'] = max(0, math_score - 1.5)
    elif strength == "Arts & Humanities":
        scores['HS_Humanities_Score'] = min(10, math_score + 2.5)
        scores['HS_STEM_Score'] = max(0, math_score - 2.5)
    
    return scores

# --- 3. HEADER SECTION ---
col_head1, col_head2 = st.columns([1, 5])
with col_head1:
    # Use a placeholder image or remove if no logo available locally
    # st.image("logo.png", width=80) 
    pass
with col_head2:
    st.markdown("## Student Success Command Center")
    st.markdown("**AI-Powered Retention & Intervention System** | v3.0 Production Build")

st.markdown("---")

# --- 4. NAVIGATION ---
tab_overview, tab_simulator, tab_research, tab_ethics = st.tabs([
    "Executive Overview", 
    "Scenario Simulator", 
    "Research Findings", 
    "Model Governance"
])

# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW (THE STORY ARC)
# ==============================================================================
with tab_overview:
    if df is not None:
        # ROW 1: KPI CARDS
        c1, c2, c3, c4 = st.columns(4)
        
        total = len(df)
        grad_rate = len(df[df['Student Status'] == 'Graduated']) / total
        risk_rate = len(df[df['Is_Struggling'] == 1]) / total
        honor_rate = len(df[df['Is_High_Achiever'] == 1]) / total
        
        def kpi_card(col, label, value, color):
            col.markdown(f"""
            <div class="glass-card" style="text-align: center; border-bottom: 4px solid {color}; padding: 1.5rem;">
                <div class="big-stat" style="color: {color};">{value}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
            
        kpi_card(c1, "Total Students", f"{total:,}", ASHESI_SLATE)
        kpi_card(c2, "Graduation Rate", f"{grad_rate:.1%}", "#27ae60")
        kpi_card(c3, "Freshman Risk", f"{risk_rate:.1%}", "#c0392b")
        kpi_card(c4, "High Achievers", f"{honor_rate:.1%}", ASHESI_GOLD)
        
        # ROW 2: THE NARRATIVE
        col_text, col_chart = st.columns([1, 1])
        
        with col_text:
            st.markdown("""
            <div class="glass-card" style="height: 100%;">
                <h4>The Strategic Narrative</h4>
                <div style="margin-top: 1rem;">
                    <h5 style="color: #2c3e50;">1. The Context: Profile by Major</h5>
                    <p>We analyzed incoming student strengths across three dimensions: STEM, Business, and Humanities. The chart demonstrates distinct academic profiles entering each major.</p>
                </div>
                <div style="margin-top: 1.5rem;">
                    <h5 style="color: #2c3e50;">2. The Insight: Resilience</h5>
                    <p>Contrary to expectation, <b>College Algebra</b> students (lowest track) graduate at higher rates than Calculus students. They are resilient survivors who benefit from foundational support.</p>
                </div>
                <div style="margin-top: 1.5rem;">
                    <h5 style="color: #2c3e50;">3. The Action: The Year 1 Filter</h5>
                    <p>Prediction accuracy jumps from 65% (Admissions) to <b>85%</b> (End of Freshman Year). Resources must shift from screening applicants to supporting freshmen.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with col_chart:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### Incoming Academic Profile")
            st.caption("Average Standardized High School Scores (0-10) by Major")
            
            # AGGREGATE SCORES BY MAJOR
            major_scores = df.groupby('Admissions_Major')[['HS_STEM_Score', 'HS_Business_Score', 'HS_Humanities_Score']].mean().reset_index()
            
            # Melt for Bar Chart
            major_melt = major_scores.melt(id_vars='Admissions_Major', var_name='Subject', value_name='Score')
            major_melt['Subject'] = major_melt['Subject'].replace({
                'HS_STEM_Score': 'STEM', 
                'HS_Business_Score': 'Business', 
                'HS_Humanities_Score': 'Humanities'
            })
            
            fig = px.bar(major_melt, x='Admissions_Major', y='Score', color='Subject',
                         barmode='group',
                         color_discrete_sequence=[ASHESI_SLATE, ASHESI_GOLD, ASHESI_RED])
            
            fig.update_layout(plot_bgcolor='white', margin=dict(t=20, b=10), xaxis_title="", yaxis_title="Avg Score", showlegend=True, legend=dict(orientation="h", y=1.1))
            st.plotly_chart(fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 2: THE ORACLE (SCENARIO SIMULATOR)
# ==============================================================================
with tab_simulator:
    st.markdown("""
    <div style="margin-bottom: 2rem;">
        <h3>Prescriptive Analytics Engine</h3>
        <p>Configure a student profile to generate a real-time risk assessment.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col_controls, col_dashboard = st.columns([1, 2])
    
    # --- LEFT SIDE: CONTROLS ---
    with col_controls:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        with st.form("sim_form"):
            st.markdown("#### Academic Profile")
            hs_score = st.slider("Math Proficiency (0-10)", 0.0, 10.0, 7.0)
            strength = st.selectbox("Dominant Subject Area", ["STEM & Science", "Business & Economics", "Arts & Humanities"])
            gap_years = st.number_input("Gap Years (0-5)", 0, 5, 0)
            
            st.markdown("#### Demographics")
            gender = st.selectbox("Gender", ["Male", "Female"])
            aid = st.selectbox("Financial Aid Status", ["No Financial Aid", "Scholarship", "Partial Aid"])
            
            st.markdown("#### University Path")
            track = st.selectbox("Assigned Math Track", ["Calculus", "Pre-Calculus", "College Algebra"])
            major = st.selectbox("Intended Major", ["Computer Science", "MIS", "Business Admin", "Engineering"])
            
            st.markdown("---")
            st.markdown("#### Year 1 Performance (Hypothetical)")
            sim_gpa = st.number_input("Projected Freshman GPA", 0.0, 4.0, 3.0)
            
            run_btn = st.form_submit_button("Run Analysis")
        st.markdown('</div>', unsafe_allow_html=True)

    # --- RIGHT SIDE: RESULTS COCKPIT ---
    with col_dashboard:
        if run_btn and 'early_warning' in models_pkg:
            # 1. PREPARE VECTOR
            pkg = models_pkg['early_warning']
            enc = pkg['encoders']
            
            # Infer advanced scores
            adv_scores = get_advanced_features(hs_score, strength)
            
            # Encode
            vals = [
                hs_score, adv_scores['HS_STEM_Score'], adv_scores['HS_Business_Score'], adv_scores['HS_Humanities_Score'],
                gap_years,
                safe_encode(enc['track'], track),
                safe_encode(enc['gender'], gender),
                safe_encode(enc['aid'], aid),
                safe_encode(enc['major'], major)
            ]
            vec = np.array([vals])
            
            # 2. PREDICT
            prob_risk = models_pkg['early_warning']['model'].predict_proba(vec)[0][1]
            prob_delay = models_pkg['timeline']['model'].predict_proba(vec)[0][1]
            
            # 3. DISPLAY GAUGES
            c1, c2 = st.columns(2)
            
            with c1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                fig_risk = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prob_risk * 100,
                    title = {'text': "At-Risk Probability"},
                    gauge = {
                        'axis': {'range': [0, 100]},
                        'bar': {'color': ASHESI_RED if prob_risk > 0.5 else "#27ae60"},
                        'steps': [
                            {'range': [0, 50], 'color': "#f8f9fa"},
                            {'range': [50, 100], 'color': "#fff5f5"}],
                    }
                ))
                fig_risk.update_layout(height=200, margin=dict(t=30,b=10,l=20,r=20))
                st.plotly_chart(fig_risk, use_container_width=True)
                
                if prob_risk > 0.5:
                    st.error(f"High Risk Detected ({prob_risk:.1%}). Intervention recommended.")
                else:
                    st.success(f"Safe Zone ({1-prob_risk:.1%} success probability).")
                st.markdown('</div>', unsafe_allow_html=True)
                
            with c2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                fig_time = go.Figure(go.Indicator(
                    mode = "number+gauge",
                    value = prob_delay * 100,
                    title = {'text': "Delayed Graduation Risk"},
                    gauge = {
                        'shape': "bullet",
                        'axis': {'range': [0, 100]},
                        'bar': {'color': ASHESI_GOLD},
                        'threshold': {'line': {'color': "red", 'width': 2}, 'thickness': 0.75, 'value': 50}
                    }
                ))
                fig_time.update_layout(height=200, margin=dict(t=30,b=10,l=20,r=20))
                st.plotly_chart(fig_time, use_container_width=True)
                
                if prob_delay > 0.5:
                    st.warning("High probability of >4 year completion.")
                else:
                    st.info("Standard 4-year timeline projected.")
                st.markdown('</div>', unsafe_allow_html=True)

            # 4. YEAR 2 PROJECTION
            st.markdown('<div class="glass-card" style="border-left: 4px solid #2c3e50;">', unsafe_allow_html=True)
            st.markdown("#### Future Projection (End of Year 1)")
            if 'year1_success' in models_pkg:
                # Add Sim Data [GPA, FailCount] to vector
                fail_flag = 0 if sim_gpa >= 2.0 else 1
                vec_y1 = np.append(vec[0], [sim_gpa, fail_flag]).reshape(1, -1)
                
                is_high = models_pkg['year1_success']['model'].predict(vec_y1)[0]
                
                if is_high == 1:
                    st.markdown(f"If this student achieves a **{sim_gpa} GPA** in Year 1, they are projected to be a **High Achiever (Honors)**.")
                else:
                    st.markdown(f"Even with a **{sim_gpa} GPA**, the model predicts a **Standard Graduation** (No Honors). Early gaps may limit the ceiling.")
            st.markdown('</div>', unsafe_allow_html=True)

        elif run_btn:
            st.error("Model files not found. Please run the notebook to generate .pkl files.")
        else:
            st.info("Enter details on the left to run the simulation.")

# ==============================================================================
# TAB 3: RESEARCH FINDINGS
# ==============================================================================
with tab_research:
    st.markdown("#### Evidence-Based Insights")
    
    if df is not None:
        t1, t2 = st.tabs(["Curriculum Impact", "Misconduct Analysis"])
        
        with t1:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                fig = px.box(df, x='Math_Track', y='First_Year_GPA', color='Math_Track',
                             color_discrete_sequence=ASHESI_PALETTE)
                fig.update_layout(plot_bgcolor='white', title="First Year GPA by Math Track")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col_b:
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown("**Key Finding**")
                st.write("Calculus students maintain higher GPAs statistically. However, longitudinal data shows College Algebra students have high survival rates in CS, despite lower initial grades.")
                st.markdown('</div>', unsafe_allow_html=True)
                
        with t2:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                mis_data = df.groupby('Financial_Aid_Status')['Has_Academic_Case'].mean().reset_index()
                fig = px.bar(mis_data, x='Financial_Aid_Status', y='Has_Academic_Case', 
                             title="Misconduct Rate by Aid Status",
                             color_discrete_sequence=[ASHESI_SLATE])
                fig.update_layout(plot_bgcolor='white', yaxis_tickformat=".1%")
                st.plotly_chart(fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            with col_b:
                st.markdown('<div class="insight-box">', unsafe_allow_html=True)
                st.markdown("**Ethical Finding**")
                st.write("Models failed to predict misconduct based on background (Accuracy ~55%). This is a positive ethical finding: misconduct is situational, not demographic.")
                st.markdown('</div>', unsafe_allow_html=True)

# ==============================================================================
# TAB 4: GOVERNANCE
# ==============================================================================
with tab_ethics:
    st.markdown("#### Model DNA & Governance")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### Feature Importance (Top Drivers)")
        if 'early_warning' in models_pkg:
            model = models_pkg['early_warning']['model']
            feats = ['HS Math', 'HS STEM', 'HS Bus', 'HS Hum', 'Gap Years', 'Track', 'Gender', 'Aid', 'Major']
            
            if len(model.feature_importances_) == len(feats):
                imp = pd.DataFrame({'Feature': feats, 'Importance': model.feature_importances_}).sort_values('Importance')
                fig = px.bar(imp, x='Importance', y='Feature', orientation='h', color_discrete_sequence=[ASHESI_RED])
                fig.update_layout(plot_bgcolor='white', height=300)
                st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    with c2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("##### Ethical Guardrails")
        st.write("1. **Gap Year Handling:** Gap years are capped at 5 years to prevent outlier bias against mature students.")
        st.write("2. **Fairness Auditing:** Gender and Aid are included variables but show <5% feature importance, confirming the model relies on academic history, not demographics.")
        st.write("3. **Human-in-the-Loop:** These scores are probabilistic flags for discussion, not automated decisions.")
        st.markdown('</div>', unsafe_allow_html=True)