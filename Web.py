# app.py
import streamlit as st
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pathlib import Path

# 1. 加载数据
data_path = Path(r"F:\数据\（1）机器学习预测模型数据\A012-CHARLS 2011-2020年已清洗最全"
                 r"\charls2011~2020清洗好+原版数据\整理完的-charls数据\Lasso回归\data_selected.xlsx")
data = pd.read_excel(data_path)

selected_features = [
    'age', 'gender', 'familysize', 'exercise', 'totmet',
    'srh', 'diabe', 'cancre', 'hearte', 'satlife',
    'iadl', 'pain'
]
X = data[selected_features]
y = data['hibpe']

# 2. 划分数据并训练模型
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=123
)
model = xgb.XGBClassifier(
    learning_rate=0.1, max_depth=3, n_estimators=100, subsample=0.8
)
model.fit(X_train, y_train)

# 3. 缓存 Explainer，参数名前加 "_"，避免 UnhashableParamError
@st.cache_resource
def load_model_and_explainer(_model):
    explainer = shap.TreeExplainer(_model)
    return _model, explainer

# 函数定义结束后，在顶格调用
model, explainer = load_model_and_explainer(model)

# —— Streamlit 页面布局 —— #
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
    # 构造新样本并预测
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
