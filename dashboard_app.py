import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# === Carregamento ===
st.set_page_config(page_title="ValidaCursos - Dashboard", layout="wide")

st.title("📊 Dashboard Interativo — ValidaCursos")
st.markdown("Explore os dados e visualize as previsões do modelo treinado.")

DATA_PATH = "data/uploads"
MODEL_PATH = "models/store"

# Dataset mais recente
import os
csvs = sorted([f for f in os.listdir(DATA_PATH) if f.endswith(".csv")])
if not csvs:
    st.warning("Nenhum dataset encontrado. O administrador deve fazer upload.")
    st.stop()

csv_file = os.path.join(DATA_PATH, csvs[-1])
df = pd.read_csv(csv_file)
st.success(f"Dataset carregado: `{csv_file}` — {df.shape[0]} linhas")

# Modelo mais recente
models = sorted([f for f in os.listdir(MODEL_PATH) if f.endswith(".joblib")])
if not models:
    st.warning("Nenhum modelo treinado encontrado. O administrador deve treinar o modelo.")
    st.stop()

model_file = os.path.join(MODEL_PATH, models[-1])
model_meta = joblib.load(model_file)
pipe = model_meta["pipeline"]
target = model_meta["target"]
features = model_meta["features"]

st.sidebar.header("🔧 Filtros e Parâmetros")

# === Filtros ===
cols = df.columns.tolist()
cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()
num_cols = df.select_dtypes(include=["number"]).columns.tolist()

filtro_col = st.sidebar.selectbox("Filtrar por coluna categórica:", cat_cols)
valor_sel = st.sidebar.selectbox("Valor:", df[filtro_col].unique())

df_filt = df[df[filtro_col] == valor_sel]

# === Gráficos ===
col1, col2 = st.columns(2)

with col1:
    cat_col = st.selectbox("Gráfico de barras — coluna categórica", cat_cols)
    fig_bar = px.bar(df_filt[cat_col].value_counts(), title=f"Distribuição — {cat_col}")
    st.plotly_chart(fig_bar, use_container_width=True)

with col2:
    if len(num_cols) >= 2:
        x = st.selectbox("Eixo X", num_cols, index=0)
        y = st.selectbox("Eixo Y", num_cols, index=1)
        color = st.selectbox("Colorir por", cat_cols)
        fig_disp = px.scatter(df_filt, x=x, y=y, color=color, title="Dispersão")
        st.plotly_chart(fig_disp, use_container_width=True)

# === Predição Interativa ===
st.header("🔮 Previsão com o modelo treinado")

col_input = {}
for f in features:
    if f in cat_cols:
        col_input[f] = st.selectbox(f, sorted(df[f].dropna().unique()))
    else:
        col_input[f] = st.slider(f, float(df[f].min()), float(df[f].max()), float(df[f].mean()))

if st.button("Gerar previsão"):
    pred = pipe.predict(pd.DataFrame([col_input]))[0]
    st.success(f"**Resultado previsto ({target}): {pred:.2f}**")
