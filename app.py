import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Placement Predictor", page_icon="🎓", layout="centered")

model = pickle.load(open("Placement_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

.stApp{
    background: linear-gradient(135deg, #1a0033, #330066, #4d0099);
    font-family: 'Poppins', sans-serif;
}

.main.block-container {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
}


h1 {
    text-align: center;
    font-size: 50px;
    font-weight: 700;
    color: #00F0FF;
    text-shadow: 0 0 10px #00F0FF, 0 0 20px #00F0FF, 0 0 30px #00F0FF;
    margin-bottom: 5px;
}


.subtitle {
    text-align: center;
    color: #E0E0FF;
    font-size: 18px;
    font-weight: 400;
    margin-bottom: 30px;
}

div.stButton > button {
    background: transparent;
    color: #FF00FF;
    border: 2px solid #FF00FF;
    border-radius: 12px;
    height: 55px;
    width: 100%;
    font-size: 20px;
    font-weight: 700;
    transition: all 0.3s ease;
    box-shadow: 0 0 10px #FF00FF;
}
div.stButton > button:hover {
    background: #FF00FF;
    color: #1a0033;
    box-shadow: 0 0 20px #FF00FF, 0 0 40px #FF00FF;
    transform: scale(1.02);
}


div[data-testid="stNumberInput"] label {
    color: #39FF14!important;
    font-size: 20px;
    font-weight: 600;
    text-shadow: 0 0 5px #39FF14;
}


div[data-testid="stNumberInput"] input {
    background: rgba(255, 255, 255, 0.1);
    color: #FFFFFF;
    font-size: 18px;
    font-weight: 600;
    border: 2px solid #00F0FF;
    border-radius: 10px;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #2D1B69, #4C1D95, #6D28D9);
    border-right: 3px solid #A78BFA;
}
[data-testid="stSidebar"] * {
    color: #F5F3FF!important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #C4B5FD!important;
    font-weight: 700;
}
[data-testid="stSidebar"] hr {
    border-color: #A78BFA;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<h1>⚡ Placement Predictor AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Neural Network powered career forecasting</p>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 📌 Mission Control")
    st.write("Predict placement outcomes using advanced ML algorithms.")

    st.markdown("---")
    st.markdown("## 📊 Model Specs")
    st.write("""
    **Algorithm:** Logistic Regression
    **Framework:** Scikit-learn
    \n**Input Features:**
    \n- 🕜 Hours Studied
    \n- 📝 Test Score
    """)

    st.markdown("---")
    st.markdown("## 🛠️ Tech Stack")
    st.write("- Python 3.11\n- Streamlit\n- Scikit-learn\n- Pandas\n- NumPy")

    st.markdown("---")
    st.markdown("## 🎯 How to Use")
    st.write("1. Input study hours\n2. Input test score\n3. Hit Predict")

    st.markdown("---")
    st.markdown("## 👩🏻‍💻 Creator")
    st.write("**Himangi Gupta**")
    st.markdown("[🔗 GitHub](https://github.com/kanak2349299)")

    st.markdown("---")
    st.info("💡 For educational demonstration only")

col1, col2 = st.columns(2, gap="large")

with col1:
    hours = st.number_input("📚 Hours Studied", min_value=0.0, max_value=24.0, value=6.0, step=0.5)

with col2:
    score = st.number_input("📊 Test Score", min_value=0.0, max_value=100.0, value=75.0, step=1.0)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🚀 INITIATE PREDICTION"):
    data = scaler.transform(np.array([[hours, score]]))
    prediction = model.predict(data)
    prob = model.predict_proba(data)[0][1]

    st.markdown("<br>", unsafe_allow_html=True)

    if prediction[0] == 1:
        st.success(f"🎉 PLACEMENT CONFIRMED!")
        st.metric(label="📈 Success Probability", value=f"{prob*100:.1f}%")
        st.balloons()
        st.markdown(f"""
        <div style="background: rgba(57, 255, 20, 0.15); border: 2px solid #39FF14; padding:20px; border-radius:15px; color:#39FF14; text-align:center;">
        <h2>ACCESS GRANTED ✅</h2>
        <p>High probability of placement. Keep grinding!</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("❌ PLACEMENT DENIED")
        st.metric(label="📈 Success Probability", value=f"{prob*100:.1f}%")
        st.markdown(f"""
        <div style="background: rgba(255, 0, 255, 0.15); border: 2px solid #FF00FF; padding:20px; border-radius:15px; color:#FF00FF; text-align:center;">
        <h2>ACCESS DENIED ⚠️</h2>
        <p>Increase study hours >8 and score >80 for better results.</p>
        </div>
        """, unsafe_allow_html=True)