import streamlit as st
import pandas as pd
import joblib
import requests
import io

st.set_page_config(page_title="Statistical Test Selector", layout="centered")

# -------------------------------
# 1. Загрузка модели и энкодера из GitHub
# -------------------------------

@st.cache_resource
def load_artifacts():
    # замените <USERNAME>/<REPO> путём к вашему репозиторию
    base_url = "https://raw.githubusercontent.com/sss0411/STS/main/artifacts/"

    model_url = base_url + "stat_test_model.joblib"
    encoder_url = base_url + "encoder.joblib"

    model_bytes = requests.get(model_url).content
    encoder_bytes = requests.get(encoder_url).content

    model = joblib.load(io.BytesIO(model_bytes))
    encoder = joblib.load(io.BytesIO(encoder_bytes))
    return model, encoder

model, encoder = load_artifacts()


# -------------------------------
# 2. Интерфейс Streamlit
# -------------------------------
st.title("🔬 Statistical Test Selector (STS)")
st.write("Введите параметры вашей исследовательской задачи:")

description = st.text_area("Description (optional)", height=100)

variables = st.text_input("Variables (пример: 'Glucose; Treatment')")

variable_types = st.selectbox("Variable Types:", [
    "continuous",
    "categorical",
    "continuous + categorical",
    "categorical + categorical",
    "time-to-event",
    "mixed"
])

num_groups = st.number_input("Number of Groups:", min_value=1, max_value=20, value=2)

paired = st.selectbox("Paired or Independent:", [
    "independent",
    "paired",
    "unknown"
])

normality = st.selectbox("Normality:", [
    "yes",
    "no",
    "unknown"
])

outcome_type = st.selectbox("Outcome Type:", [
    "continuous",
    "categorical",
    "time-to-event",
    "other"
])


# -------------------------------
# 3. Проверка unsupported cases (логика из вашего алгоритма)
# -------------------------------

def check_unsupported(num_groups, paired, outcome_type, variable_types):
    # Парные группы >2
    if outcome_type == "continuous" and num_groups > 2 and paired == "paired":
        return True

    # Time-to-event = survival → не поддерживается
    if "time-to-event" in variable_types or outcome_type == "time-to-event":
        return True

    # Смешанные типы переменных
    if variable_types == "mixed":
        return True

    return False


# -------------------------------
# 4. Кнопка "Предсказать"
# -------------------------------
if st.button("Recommend Statistical Test"):
    
    if check_unsupported(num_groups, paired, outcome_type, variable_types):
        st.error("❌ Sorry, this model supports only basic classical statistical tests.")
    else:
        # Подготовка строки
        input_df = pd.DataFrame([{
            'Variables': variables,
            'Variable Types': variable_types,
            'Number of Groups': num_groups,
            'Paired or Independent': paired,
            'Normality': normality,
            'Outcome Type': outcome_type
        }])

        # OneHotEncoder
        X_enc = encoder.transform(input_df.astype(str))

        # Предсказание
        pred = model.predict(X_enc)[0]

        st.success(f"### ✅ Recommended test: **{pred}**")

        # Показываем описание (если есть)
        if description.strip():
            st.write("### Research Question")
            st.info(description)

        st.write("---")
        st.write("### Input summary")
        st.json(input_df.to_dict(orient='records')[0])
