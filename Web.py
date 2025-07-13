# web.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 1. 加载预训练模型
model = joblib.load('xgb_model.pkl')

# 2. 定义特征名称（与你训练时用的一致）
feature_names = [
    'age', 'gender', 'familysize', 'exercise', 'totmet',
    'srh', 'diabe', 'cancre', 'hearte', 'satlife',
    'iadl', 'pain'
]

# 3. Streamlit 界面
st.title("Predicting Hypertension in Patients with Chronic Diseases")

# Age
age = st.number_input(
    "Age:", min_value=18, max_value=120, value=60
)

# Gender
gender = st.selectbox(
    "Gender (0=Female, 1=Male):",
    options=[0, 1],
    format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)'
)

# Family size
familysize = st.number_input(
    "Family size:", min_value=1, max_value=20, value=3
)

# Exercise
exercise = st.selectbox(
    "Exercise (0=No, 1=Yes):",
    options=[0, 1],
    format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)'
)

# Total metabolism
totmet = st.number_input(
    "Total metabolism:", min_value=0.0, step=0.1, value=5.0
)

# Self‐rated health
srh = st.number_input(
    "Self-rated health (1=Poor, 5=Excellent):",
    min_value=1, max_value=5, value=3
)

# Diabetes
diabe = st.selectbox(
    "Diabetes (0=No, 1=Yes):",
    options=[0, 1],
    format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)'
)

# Cancer
cancre = st.selectbox(
    "Cancer (0=No, 1=Yes):",
    options=[0, 1],
    format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)'
)

# Heart disease
hearte = st.selectbox(
    "Heart disease (0=No, 1=Yes):",
    options=[0, 1],
    format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)'
)

# Life satisfaction
satlife = st.number_input(
    "Life satisfaction (1=Low, 5=High):",
    min_value=1, max_value=5, value=3
)

# IADL score
iadl = st.number_input(
    "IADL score:", min_value=0.0, max_value=10.0, step=0.1, value=2.0
)

# Pain
pain = st.selectbox(
    "Pain (0=No, 1=Yes):",
    options=[0, 1],
    format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)'
)

# 4. 组织成模型输入
feature_values = [
    age, gender, familysize, exercise, totmet,
    srh, diabe, cancre, hearte, satlife,
    iadl, pain
]
features = np.array([feature_values])

# 5. 点击 Predict 后执行预测、SHAP 并展示
if st.button("Predict"):
    # 预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # 显示结果
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 建议文字
    prob = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a higher risk of health issues. "
            f"The model predicts that your probability is {prob:.1f}%. "
            "It is recommended to consult with your healthcare provider for further evaluation."
        )
    else:
        advice = (
            f"According to our model, you have a lower risk of health issues. "
            f"The model predicts that your probability is {prob:.1f}%. "
            "Maintaining a healthy lifestyle is still important."
        )
    st.write(advice)

    # 计算 SHAP 并用 matplotlib 静态力图
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(
        pd.DataFrame([feature_values], columns=feature_names)
    )
    fig, ax = plt.subplots(figsize=(8, 2))
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        pd.DataFrame([feature_values], columns=feature_names),
        matplotlib=True,
        show=False
    )
    plt.tight_layout()
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=300)
    st.image("shap_force_plot.png")
