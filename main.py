# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# 1. Introdução
# =========================================
st.title("Análise Meteorologia e Qualidade do Ar em Portugal")
st.markdown("Este dashboard mostra os datasets, EDA e Machine Learning de forma limpa.")

# =========================================
# 2. Carregamento dos Dados (sem APIs)
# =========================================
st.header("📂 Carregamento dos Dados")

df_meteo = pd.read_csv("dataset_meteorologico_portugal.csv")
df_meteo["date"] = pd.to_datetime(df_meteo["date"])

dfqualidadear = pd.read_excel("QualidadeAr2.xlsx")
dfqualidadear2023 = pd.read_excel("Qualar2023.xlsx")

st.subheader("Meteorologia")
st.dataframe(df_meteo.head())

st.subheader("Qualidade do Ar 2025")
st.dataframe(dfqualidadear.head())

st.subheader("Qualidade do Ar 2023")
st.dataframe(dfqualidadear2023.head())

# =========================================
# 3. Limpeza e Transformação
# =========================================
st.header("🧹 Limpeza e Transformação")

# Exemplo: converter datas e calcular médias semanais
dfqualidadear['Data'] = pd.to_datetime(dfqualidadear['Data'], dayfirst=True)
dfqualidadear['Semana'] = dfqualidadear['Data'].dt.isocalendar().week
dfqualidadear['Ano'] = dfqualidadear['Data'].dt.year

poluentes = ['C6H6', 'CO', 'NO2', 'O3', 'PM2.5', 'PM10', 'SO2']
for p in poluentes:
    dfqualidadear[p] = pd.to_numeric(dfqualidadear[p], errors='coerce')

st.write("Após limpeza:")
st.dataframe(dfqualidadear.head())

# =========================================
# 4. EDA
# =========================================
st.header("📊 Análise Exploratória (EDA)")

# Boxplots
fig, axes = plt.subplots(1, len(poluentes), figsize=(20,5))
for i, col in enumerate(poluentes):
    sns.boxplot(y=dfqualidadear[col], ax=axes[i], color="skyblue")
    axes[i].set_title(col)
st.pyplot(fig)

# Comparação 2023 vs 2025
st.subheader("Comparação 2023 vs 2025")
dfqualidadear2023['Data-Hora'] = pd.to_datetime(dfqualidadear2023['Data-Hora'])
dfqualidadear2023['Ano'] = dfqualidadear2023['Data-Hora'].dt.year
st.dataframe(dfqualidadear2023.head())

# =========================================
# 5. Machine Learning
# =========================================
st.header("🤖 Machine Learning")

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Exemplo simples: prever PM10 com meteorologia
df_ml = df_meteo.dropna(subset=["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_80m"])
X = df_ml[["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_80m"]]
y = df_ml["temperature_2m_max"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

st.write("Erro quadrático médio:", mean_squared_error(y_test, y_pred))
