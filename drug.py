import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Drug Response Predictor",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Simple, clean CSS
st.markdown("""
<style>
.stApp {
    background: #f5f6fa !important;
}
h1, h2, h3 {
    color: #4a148c !important;
    font-weight: 700;
}
.stButton>button {
    background: #7e57c2;
    color: white;
    border-radius: 10px;
    font-weight: 600;
    transition: background 0.3s;
}
.stButton>button:hover {
    background: #4a148c;
}
[data-testid="stSidebar"] {
    background: #ede7f6;
}
[data-testid="stMetric"] {
    background: #ede7f6;
    border-radius: 12px;
}
.stDataFrame {
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

USERS_FILE = "users.csv"

def load_users():
    if not os.path.exists(USERS_FILE):
        df = pd.DataFrame([["admin", "admin123"]], columns=["username", "password"])
        df.to_csv(USERS_FILE, index=False)
    return pd.read_csv(USERS_FILE)

def add_user(username, password):
    df = load_users()
    if username in df['username'].values:
        return False
    df = pd.concat([df, pd.DataFrame({"username": [username], "password": [password]})], ignore_index=True)
    df.to_csv(USERS_FILE, index=False)
    return True

def login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("<h1 style='text-align: center;'>🧬 Drug Response Predictor</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #7e57c2;'>Predict cancer cell line sensitivity using ML</p>", unsafe_allow_html=True)
        st.markdown("---")

        page = st.radio("", ["Login", "Register"], horizontal=True)
        users = load_users()

        if page == "Login":
            username = st.text_input("👤 Username", placeholder="Enter your username")
            password = st.text_input("🔑 Password", type="password", placeholder="Enter your password")
            if st.button("Login", use_container_width=True):
                row = users[(users.username == username) & (users.password == password)]
                if len(row):
                    st.session_state.authenticated = True
                    st.session_state.username = username
                    st.success(f"✨ Welcome back, {username}!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials")
        else:
            newuser = st.text_input("👤 Choose Username", placeholder="Minimum 3 characters")
            newpass = st.text_input("🔑 Choose Password", type="password", placeholder="Minimum 4 characters")
            if st.button("Register", use_container_width=True):
                if len(newuser) < 3 or len(newpass) < 4:
                    st.warning("⚠️ Username or password too short")
                elif add_user(newuser, newpass):
                    st.success("✅ Registration successful! Please login.")
                    st.balloons()
                else:
                    st.error("❌ Username already exists")

        st.markdown("---")
        st.info("💡 **Demo:** Use `demo` / `demo123` to try the app")

def logout():
    st.session_state.authenticated = False
    st.session_state.username = None
    st.rerun()

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False
if "username" not in st.session_state:
    st.session_state.username = None

def main_app():
    st.markdown(f"<h1 style='text-align: center;'>🧬 Cancer Cell Line Drug Response Prediction</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center; color: #7e57c2;'>Welcome, <strong>{st.session_state['username']}</strong> | {datetime.now().strftime('%H:%M:%S')}</p>", unsafe_allow_html=True)

    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/000000/dna.png", width=80)
        st.markdown("### 📋 Navigation")
        page = st.radio("", ["🏠 Home", "🔬 Predict", "📊 Model Info", "📁 Example Data"], label_visibility="collapsed")
        st.markdown("---")
        st.markdown("### 👥 About")
        st.info(
            "**Authors:**\n"
            "- Dhanya Shetty (24251202)\n"
            "- Ashritha K (24251208)\n\n"
            "**Institution:**\n"
            "St Aloysius University"
        )
        if st.button("🚪 Logout", use_container_width=True):
            logout()

    if page == "🏠 Home":
        home_page()
    elif page == "🔬 Predict":
        predict_page()
    elif page == "📊 Model Info":
        model_info_page()
    elif page == "📁 Example Data":
        example_data_page()

def home_page():
    st.markdown("## 👋 Welcome to the Drug Response Prediction System")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background: #ede7f6;
                    padding: 2rem; border-radius: 15px; 
                    box-shadow: 0 8px 16px rgba(126, 87, 194, 0.1);'>
            <h3 style='color: #4a148c;'>🎯 Project Overview</h3>
            <p>This application predicts cancer cell line sensitivity to anticancer drugs using gene expression profiles from the GDSC2 database.</p>
            <ul>
                <li>🧬 Gene expression-based prediction</li>
                <li>🤖 ML models (SVM, RF, XGBoost)</li>
                <li>📈 64% accuracy with SVM</li>
                <li>💊 5 anticancer drugs supported</li>
                <li>🔬 389 cancer cell lines</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("### 📊 Quick Stats")
        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("🤖 Models", "3", "SVM, RF, XGB", delta_color="off")
            st.metric("🧬 Cell Lines", "389", "Common")
        with metric_col2:
            st.metric("✅ Best Accuracy", "64.0%", "SVM Model")
            st.metric("🔬 Features", "500", "Top variance genes")
    st.markdown("---")
    st.markdown("""
    <div style='background: #fff; padding: 2rem; border-radius: 15px; border: 2px solid #d1c4e9;'>
        <h3 style='color: #4a148c;'>🚀 How to Use</h3>
        <ol style='font-size: 16px; line-height: 2; color: #4a148c;'>
            <li><strong>Navigate to Predict</strong> page from sidebar</li>
            <li><strong>Upload CSV file</strong> with gene expression data</li>
            <li><strong>Select drug</strong> from dropdown menu</li>
            <li><strong>View predictions</strong> with probabilities</li>
            <li><strong>Download results</strong> as CSV file</li>
        </ol>
        <p style='margin-top: 1rem; color: #7e57c2;'>
            💡 Check <strong>Example Data</strong> page for file format!
        </p>
    </div>
    """, unsafe_allow_html=True)

def predict_page():
    st.markdown("## 🔬 Drug Response Prediction")
    st.markdown("Upload your gene expression matrix and get instant predictions!")
    @st.cache_resource
    def load_model_files():
        with open('best_model_svm.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('top_genes.pkl', 'rb') as f:
            top_genes = pickle.load(f)
        return model, scaler, top_genes
    model, scaler, top_genes = load_model_files()
    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"], help="Must include CELL_LINE_NAME and all gene columns")
    with col2:
        drug = st.selectbox("💊 Select Drug", ['Oxaliplatin', 'Ulixertinib', 'Fulvestrant', 'Selumetinib', 'Dactinomycin'])
    confidence = st.slider("🎯 Confidence Threshold", 0.0, 1.0, 0.5, 0.01, help="Filter predictions above this confidence level")
    if uploaded_file is not None:
        try:
            with st.spinner("🔄 Processing your data..."):
                data = pd.read_csv(uploaded_file)
                with st.expander("👀 File Preview"):
                    st.dataframe(data.head(), use_container_width=True)
                missing_cols = [col for col in top_genes if col not in data.columns]
                if missing_cols:
                    st.error(f"❌ Missing {len(missing_cols)} gene columns")
                    return
                if "CELL_LINE_NAME" not in data.columns:
                    st.error("❌ Missing CELL_LINE_NAME column!")
                    return
                input_X = data[top_genes].values
                X_scaled = scaler.transform(input_X)
                predictions = model.predict(X_scaled)
                probas = model.predict_proba(X_scaled)
                confidence_scores = probas.max(axis=1)
                result_df = pd.DataFrame({
                    "CELL_LINE_NAME": data["CELL_LINE_NAME"],
                    "Drug": drug,
                    "Prediction": ["Sensitive" if pred == 1 else "Resistant" for pred in predictions],
                    "Confidence": confidence_scores,
                    "Prob_Resistant": probas[:, 0],
                    "Prob_Sensitive": probas[:, 1]
                })
                filtered = result_df[result_df["Confidence"] >= confidence]
                st.success(f"✅ Predictions complete! {len(filtered)}/{len(result_df)} samples pass threshold")
                col1, col2, col3 = st.columns(3)
                with col1:
                    sens_count = (filtered['Prediction'] == 'Sensitive').sum()
                    st.metric("✅ Sensitive", sens_count, f"{100*sens_count/len(filtered):.1f}%")
                with col2:
                    res_count = (filtered['Prediction'] == 'Resistant').sum()
                    st.metric("❌ Resistant", res_count, f"{100*res_count/len(filtered):.1f}%")
                with col3:
                    avg_conf = filtered['Confidence'].mean()
                    st.metric("🎯 Avg Confidence", f"{avg_conf:.2%}")
                st.dataframe(filtered, use_container_width=True, height=400)
                st.download_button(
                    "📥 Download Results",
                    filtered.to_csv(index=False),
                    f"{drug}_predictions.csv",
                    "text/csv",
                    use_container_width=True
                )
                fig = px.bar(
                    filtered,
                    x="CELL_LINE_NAME",
                    y="Prob_Sensitive",
                    color="Prediction",
                    color_discrete_map={"Sensitive": "#4CAF50", "Resistant": "#f44336"},
                    title=f"Sensitivity Probability for {drug}",
                    labels={"Prob_Sensitive": "Probability of Sensitivity", "CELL_LINE_NAME": "Cell Line"}
                )
                fig.update_layout(
                    plot_bgcolor='#fff',
                    paper_bgcolor='#fff',
                    font=dict(color='#4a148c'),
                    showlegend=True,
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Error: {e}")

def model_info_page():
    st.markdown("## 📊 Model Information")
    tab1, tab2, tab3 = st.tabs(["📋 Methodology", "📈 Performance", "💊 Drugs"])
    with tab1:
        st.markdown("""
### 🔬 Methodology

**Data Source:** GDSC2 (Genomics of Drug Sensitivity in Cancer)

**Dataset Details:**
- 389 common cancer cell lines
- Gene expression data (log2 transformed)
- IC50 drug response values

**Preprocessing:**
1. Log2 transformation
2. Top 500 genes by variance
3. Standard scaling (mean=0, std=1)
4. Binary classification (Sensitive vs Resistant)

**Models:**
- Support Vector Machine (SVM) with RBF kernel
- Random Forest Classifier
- XGBoost Classifier

**Validation:**
- 80-20 train-test split
- 5-fold cross-validation
""")
    with tab2:
        model_data = pd.DataFrame({
            'Model': ['SVM', 'XGBoost', 'Random Forest'],
            'Accuracy': [0.640, 0.623, 0.603],
            'F1-Score': [0.641, 0.623, 0.594],
            'ROC-AUC': [0.663, 0.635, 0.631]
        })
        st.dataframe(model_data, use_container_width=True)
        fig = go.Figure()
        for metric in ['Accuracy', 'F1-Score', 'ROC-AUC']:
            fig.add_trace(go.Bar(
                name=metric,
                x=model_data['Model'],
                y=model_data[metric],
                text=model_data[metric],
                textposition='auto'
            ))
        fig.update_layout(
            barmode='group',
            title='Model Performance Comparison',
            plot_bgcolor='#fff',
            paper_bgcolor='#fff',
            font=dict(color='#4a148c'),
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        st.success("🏆 **Best Model:** SVM with 64% accuracy")
    with tab3:
        drugs_info = {
            "Oxaliplatin": "Platinum-based chemotherapy drug",
            "Ulixertinib": "ERK1/2 inhibitor",
            "Fulvestrant": "Estrogen receptor antagonist",
            "Selumetinib": "MEK inhibitor",
            "Dactinomycin": "Actinomycin antibiotic"
        }
        for drug, desc in drugs_info.items():
            st.markdown(f"**💊 {drug}:** {desc}")

def example_data_page():
    st.markdown("## 📁 Example Data & Templates")
    try:
        with open('top_genes.pkl', 'rb') as f:
            top_genes = pickle.load(f)
        columns = ["CELL_LINE_NAME"] + top_genes
        template_df = pd.DataFrame(columns=columns)
        template_csv = template_df.to_csv(index=False)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div style='background: #ede7f6;
                        padding: 2rem; border-radius: 15px;
                        box-shadow: 0 8px 16px rgba(126, 87, 194, 0.1);'>
                <h3 style='color: #4a148c;'>📄 Input Template</h3>
                <p>Empty template with correct structure</p>
            </div>
            """, unsafe_allow_html=True)
            st.download_button(
                "📥 Download Template",
                template_csv,
                "input_template.csv",
                "text/csv",
                use_container_width=True
            )
        with col2:
            st.info(f"✅ Template ready with **{len(top_genes)}** gene columns!")
            st.markdown("""
            **Required columns:**
            - `CELL_LINE_NAME`
            - 500 gene expression columns
            """)
    except Exception as e:
        st.error(f"❌ Error: {e}")

if not st.session_state["authenticated"]:
    login_page()
else:
    main_app()
