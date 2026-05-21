import streamlit as st
import pandas as pd
import mlflow
import mlflow.sklearn

st.set_page_config(
    page_title="Predicción Estudiantes",
    layout="centered"
)

st.title("📚 Predicción de Aprobación de Estudiantes")
st.write(
    "Esta aplicación usa un modelo de MLflow para predecir si un estudiante aprueba o no."
)

# Conexión a MLflow
mlflow.set_tracking_uri("http://127.0.0.1:9090")

# Cambiar versión si es necesario
MODEL_URI = "models:/arboles_Estudiantes/1"

@st.cache_resource
def cargar_modelo():
    return mlflow.sklearn.load_model(MODEL_URI)

model = cargar_modelo()

st.sidebar.header("Configuración")
st.sidebar.write(f"Modelo cargado: `{MODEL_URI}`")

#Verificar que columnas espera 
#st.write("Columnas que espera el modelo:")
#st.write(model.feature_names_in_)

# ================================
# 🔹 ENTRADAS DEL USUARIO (ESTUDIANTES)
# ================================

st.subheader("Ingreso de datos del estudiante")

col1, col2 = st.columns(2)

with col1:
    carrera = st.selectbox("Carrera", ["Sistemas", "Industrial", "Civil", "Administración"])
    modalidad = st.selectbox("Modalidad", ["Presencial", "Virtual", "Híbrida"])
    beca = st.selectbox("¿Tiene beca?", ["Si", "No"])
    

with col2:
    edad = st.number_input("Edad", min_value=16, max_value=60, value=20)
    promedio = st.number_input("Promedio", min_value=0.0, max_value=10.0, value=7.0)
    asistencias = st.slider("Asistencia (%)", 0, 100, 80)

# ================================
# 🔹 DATAFRAME PARA EL MODELO
# ================================


datos = pd.DataFrame([{
    "carrera": carrera,
    "modalidad": modalidad,
    "beca": beca,
    "edad": edad,
    "promedio": promedio,
    "asistencias": asistencias,
}])

st.subheader("Datos enviados al modelo")
st.dataframe(datos)

# ================================
# 🔹 PREDICCIÓN
# ================================

if st.button("Predecir"):
    prediccion = model.predict(datos)[0]

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(datos)[0]
        prob_no = proba[0]
        prob_si = proba[1]
    else:
        prob_no = None
        prob_si = None

    if prediccion == 1:
        st.success("✅ El estudiante APRUEBA")
    else:
        st.error("❌ El estudiante NO APRUEBA")

    if prob_si is not None:
        st.write(f"Probabilidad de NO aprobar: {prob_no:.4f}")
        st.write(f"Probabilidad de aprobar: {prob_si:.4f}")

# ================================
# 🔹 NOTA FINAL
# ================================

st.caption(
    "Nota: Las variables deben coincidir EXACTAMENTE con las usadas al entrenar el modelo."
)
