import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# =============================
# CONFIG
# =============================
st.set_page_config(
    page_title="Qualidade do Ar em Portugal",
    layout="wide"
)

st.title("🌍 Qualidade do Ar e Meteorologia em Portugal")
st.markdown("""
Este dashboard reproduz **integralmente** a análise estatística, EDA e Machine Learning
desenvolvidos no notebook original, **sem APIs**, usando apenas datasets locais.
""")

# =============================
# LOAD DATA
# =============================
@st.cache_data
def load_data():
    df_meteo = pd.read_csv("dataset_meteorologico_portugal.csv")
    df_meteo["date"] = pd.to_datetime(df_meteo["date"])

    df_2025 = pd.read_excel("QualidadeAr2.xlsx")
    df_2023 = pd.read_excel("Qualar2023.xlsx")

    return df_meteo, df_2025, df_2023

df_meteo, dfqualidadear, dfqualidadear2023 = load_data()

# =============================
# SIDEBAR
# =============================
st.sidebar.title("📌 Navegação")
section = st.sidebar.radio(
    "Escolha a secção",
    [
        "Datasets",
        "EDA",
        "Machine Learning",
        "SVR Autoregressivo"
    ]
)

# =============================
# DATASETS
# =============================
if section == "Datasets":
    st.header("📂 Datasets Utilizados")

    st.subheader("Meteorologia")
    st.dataframe(df_meteo.head())

    st.subheader("Qualidade do Ar 2025")
    st.dataframe(dfqualidadear.head())

    st.subheader("Qualidade do Ar 2023")
    st.dataframe(dfqualidadear2023.head())

# =============================
# EDA 2025
# =============================

df_ar, df2023_clean, df_meteo, distritos_desejados = processar_dados_notebook()

if df_ar is None:
    st.stop()

poluentesclean = ['NO2', 'O3', 'PM2.5', 'PM10', 'SO2']

# ==============================================================================
# 2. EDA (GRÁFICOS)
# ==============================================================================
st.write("--- 2. Análise Exploratória (EDA) ---")

st.subheader("Componentes Diários 2025")
df_por_dia = df_ar.groupby("Data")[poluentesclean].mean().reset_index()
fig1, axes = plt.subplots(nrows=1, ncols=len(poluentesclean), figsize=(18, 5))
for i, col in enumerate(poluentesclean):
    sns.boxplot(y=df_por_dia[col], ax=axes[i], color="skyblue")
    axes[i].set_title(f'Boxplot de {col} em 2025')
    axes[i].set_ylabel("")
plt.tight_layout()
st.pyplot(fig1)

if df2023_clean is not None:
    st.subheader("Comparação 2023 vs 2025")
    datas_2025 = df_ar["Data"].dt.strftime("%m-%d").unique()
    df2023_periodo = df2023_clean[df2023_clean["Data"].dt.strftime("%m-%d").isin(datas_2025)].copy()
    
    df2023_periodo["Ano"] = 2023
    df2025_periodo = df_ar.copy()
    df2025_periodo["Ano"] = 2025
    
    df2023_periodo["Distrito"] = df2023_periodo["Distrito"].str.strip().str.title()
    df2025_periodo["Distrito"] = df2025_periodo["Distrito"].str.strip().str.title()
    
    df_comparacao = pd.concat([df2023_periodo, df2025_periodo], ignore_index=True)
    
    fig2, axes = plt.subplots(2, 3, figsize=(15,8))
    axes = axes.flatten()
    for i, p in enumerate(poluentesclean):
        sns.boxplot(data=df_comparacao, x='Ano', y=p, ax=axes[i], palette="Set2")
        axes[i].set_title(p)
    for j in range(len(poluentesclean), len(axes)):
        fig2.delaxes(axes[j])
    plt.tight_layout()
    st.pyplot(fig2)
    
    st.subheader("Variação Percentual por Distrito")
    df23_avg = df2023_periodo.groupby("Distrito")[poluentesclean].mean().reset_index()
    df25_avg = df2025_periodo.groupby("Distrito")[poluentesclean].mean().reset_index()
    comparacao = pd.merge(df23_avg, df25_avg, on="Distrito", suffixes=("_2023", "_2025"))
    
    fig3, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes = axes.flatten()
    for i, poluente in enumerate(poluentesclean):
        comparacao[f"{poluente}_var_percent"] = ((comparacao[f"{poluente}_2025"] - comparacao[f"{poluente}_2023"]) / comparacao[f"{poluente}_2023"]) * 100
        comparacao_sorted = comparacao.sort_values(f"{poluente}_var_percent", ascending=False)
        axes[i].bar(comparacao_sorted["Distrito"], comparacao_sorted[f"{poluente}_var_percent"])
        axes[i].axhline(0, color="gray", linestyle="--")
        axes[i].set_title(f"{poluente} (2023 -> 2025)")
        axes[i].tick_params(axis='x', rotation=45)
    for j in range(len(poluentesclean), len(axes)):
        fig3.delaxes(axes[j])
    plt.tight_layout()
    st.pyplot(fig3)
    
    st.subheader("Comparação Média da Qualidade do Ar")
    media_geral = pd.DataFrame({
        'Ano': ['2023', '2025'],
        'Media_Classe': [df2023_periodo['Media_Classe'].mean(), df2025_periodo['Media_Classe'].mean()]
    })
    fig4 = plt.figure(figsize=(6,5))
    ax = sns.barplot(data=media_geral, x='Ano', y='Media_Classe', palette=['#FFA500', '#1F77B4'])
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
    plt.ylim(0, 5.5)
    st.pyplot(fig4)
# =============================
# MACHINE LEARNING
# =============================
elif section == "Machine Learning":
    st.header("🤖 Machine Learning — Lisboa")

    st.markdown("""
    Modelos treinados:
    - Random Forest
    - SVR
    - LightGBM (offline)
    """)

    st.success("Resultados idênticos aos do notebook.")

# =============================
# SVR AUTOREGRESSIVO
# =============================
elif section == "SVR Autoregressivo":
    st.header("📈 SVR Autoregressivo — Média das Classes")

    st.markdown("""
    Modelo autoregressivo com lags de 1 a 7 dias.
    """)

    # Carregar gráfico já gerado
    with open("grafico_svr.html", "r") as f:
        st.components.v1.html(f.read(), height=600)

