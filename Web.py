# app.py
import os
import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# —— 1. 读取数据 ——
BASE_DIR = os.path.dirname(__file__)
data_path = os.path.join(BASE_DIR, "data", "data_selected.xlsx")
data = pd.read_excel(data_path)

# —— 2. 特征与标签 ——
selected_features = [
    'age', 'gender', 'familysize', 'exercise', 'totmet',
    'srh', 'diabe', 'cancre', 'hearte', 'satlife',
    'iadl', 'pain'
]
X = data[selected_features]
y = data['hibpe']

# —— 3. 划分数据集 ——
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)

# —— 4. 训练 XGBoost ——
model = xgb.XGBClassifier(
    learning_rate=0.1, max_depth=3,
    n_estimators=100, subsample=0.8
)
model.fit(X_train, y_train)

# —— 5. 缓存 SHAP Explainer ——
@st.cache_resource
def load_explainer(m):
    return shap.TreeExplainer(m)

explainer = load_explainer(model)

# —— 6. Streamlit 界面 ——
st.title("健康风险预测 (CHARLS) + SHAP 可解释性")

st.sidebar.header("输入患者特征")
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
    # 构造单条样本
    X_new = pd.DataFrame([{
        "age": age, "gender": gender, "familysize": familysize,
        "exercise": exercise, "totmet": totmet, "srh": srh,
        "diabe": diabe, "cancre": cancre, "hearte": hearte,
        "satlife": satlife, "iadl": iadl, "pain": pain
    }])
    # 预测
    proba = model.predict_proba(X_new)[0]
    pred  = model.predict(X_new)[0]

    st.subheader("预测结果")
    st.write(f"- **Predicted Class:** {pred}")
    st.write(f"- **Prediction Probabilities:** {proba}")

    # SHAP 值 & 条形图
    shap_values = explainer.shap_values(X_new)[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    shap.bar_plot(shap_values, feature_names=X_new.columns, show=False)
    plt.title("SHAP Feature Impact")
    plt.tight_layout()
    st.pyplot(fig)





