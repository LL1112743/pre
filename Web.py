# web.py
import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
from pathlib import Path
import joblib  # 或者用 pickle

# 1. 定义模型和 Explainer 的加载函数，缓存资源
@st.cache_resource
def load_model_and_explainer(_model_path: str):
    # 加载模型
    model = joblib.load(_model_path)
    # 创建 SHAP Explainer
    explainer = shap.TreeExplainer(model)
    return model, explainer

# 2. 指定模型路径（根据实际情况修改）
MODEL_PATH = Path(__file__).parent / "xgb_model.pkl"
model, explainer = load_model_and_explainer(str(MODEL_PATH))

# —— Streamlit 页面布局 —— #
st.title("CHARLS 健康风险预测 (预训练模型 + SHAP 可解释性)")

st.sidebar.header("输入患者特征")
# （以下保持和之前一致的输入控件）
age        = st.sidebar.number_input("Age",                    min_value=18,   max_value=120,   value=60)
gender     = st.sidebar.selectbox("Gender (0=F, 1=M)",        [0,1])
familysize = st.sidebar.number_input("Family size",            min_value=1,    max_value=10,    value=3)
exercise   = st.sidebar.selectbox("Exercise (0=No, 1=Yes)",   [0,1])
totmet     = st.sidebar.number_input("Total metabolism",       min_value=0.0,  step=0.1,        value=5.0)
srh        = st.sidebar.number_input("Self-rated health (1–5)",min_value=1,    max_value=5,     value=3)
diabe      = st.sidebar.selectbox("Diabetes (0=No, 1=Yes)",   [0,1])
cancre     = st.sidebar.selectbox("Cancer (0=No, 1=Yes)",     [0,1])
hearte     = st.sidebar.selectbox("Heart disease (0=No, 1=Yes)", [0,1])
satlife    = st.sidebar.number_input("Life satisfaction (1–5)",min_value=1,    max_value=5,     value=3)
iadl       = st.sidebar.number_input("IADL score",             min_value=0.0,  max_value=10.0,   step=0.1, value=2.0)
pain       = st.sidebar.selectbox("Pain (0=No, 1=Yes)",       [0,1])

if st.sidebar.button("Predict"):
    # 构造 DataFrame 并预测
    X_new = pd.DataFrame([{
        "age": age, "gender": gender, "familysize": familysize,
        "exercise": exercise, "totmet": totmet, "srh": srh,
        "diabe": diabe, "cancre": cancre, "hearte": hearte,
        "satlife": satlife, "iadl": iadl, "pain": pain
    }])
    proba = model.predict_proba(X_new)[0]
    pred  = model.predict(X_new)[0]

    st.subheader("预测结果")
    st.write(f"- **Predicted Class:** {pred}")
    st.write(f"- **Prediction Probabilities:** `{proba}`")

    # SHAP 力图
    shap_values = explainer.shap_values(X_new)
    fig, ax = plt.subplots(figsize=(8, 1.5))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X_new.iloc[0],
        matplotlib=True,
        show=False
    )
    plt.title("SHAP Force Plot for This Sample")
    plt.tight_layout()
    st.pyplot(fig)

