# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# CONFIGURAÇÃO STREAMLIT
# =========================================================
st.set_page_config(
    page_title="Qualidade do Ar em Portugal",
    layout="wide"
)

st.title("🌍 Qualidade do Ar em Portugal")
st.write("Análise Exploratória de Dados (EDA) — 2023 vs 2025")

# =========================================================
# CARREGAMENTO DOS DADOS
# =========================================================
@st.cache_data
def load_data():
    try:
        df_ar_2025 = pd.read_excel("QualidadeAr2.xlsx")
    except:
        df_ar_2025 = None

    try:
        df_ar_2023 = pd.read_excel("Qualar2023.xlsx")
    except:
        df_ar_2023 = None

    try:
        df_meteo = pd.read_csv("dataset_meteorologico_portugal.csv")
    except:
        df_meteo = None

    return df_ar_2025, df_ar_2023, df_meteo


df_ar, df2023_clean, df_meteo = load_data()

# =========================================================
# VALIDAÇÕES INICIAIS
# =========================================================
if df_ar is None or df_ar.empty:
    st.error("❌ Dataset de Qualidade do Ar 2025 não foi carregado.")
    st.stop()

df_ar["Data"] = pd.to_datetime(df_ar["Data"])

if df2023_clean is not None:
    df2023_clean["Data"] = pd.to_datetime(df2023_clean["Data"])

# Poluentes disponíveis
poluentesclean = [c for c in ["NO2", "O3", "PM2.5", "PM10", "SO2"] if c in df_ar.columns]

# =========================================================
# EDA
# =========================================================
st.write("---")
st.header("2. Análise Exploratória de Dados (EDA)")

tab1, tab2, tab3, tab4 = st.tabs([
    "📦 Componentes Diários 2025",
    "📊 Comparação 2023 vs 2025",
    "📈 Variação Percentual",
    "🌦️ Meteorologia vs Qualidade do Ar"
])

# =========================================================
# ABA 1 — COMPONENTES DIÁRIOS 2025
# =========================================================
with tab1:
    st.subheader("Componentes Diários 2025")

    df_por_dia = df_ar.groupby("Data")[poluentesclean].mean().reset_index()

    fig1, axes = plt.subplots(1, len(poluentesclean), figsize=(18, 5))
    for i, col in enumerate(poluentesclean):
        sns.boxplot(y=df_por_dia[col], ax=axes[i], color="skyblue")
        axes[i].set_title(f'Boxplot de {col} em 2025')
        axes[i].set_ylabel("")
    plt.tight_layout()
    st.pyplot(fig1)

# =========================================================
# ABA 2 — COMPARAÇÃO 2023 VS 2025
# =========================================================
with tab2:
    if df2023_clean is None or df2023_clean.empty:
        st.warning("Dataset de 2023 não disponível.")
    else:
        st.subheader("Comparação 2023 vs 2025")

        datas_2025 = df_ar["Data"].dt.strftime("%m-%d").unique()

        df2023_periodo = df2023_clean[
            df2023_clean["Data"].dt.strftime("%m-%d").isin(datas_2025)
        ].copy()

        df2023_periodo["Ano"] = 2023
        df2025_periodo = df_ar.copy()
        df2025_periodo["Ano"] = 2025

        for df_ in [df2023_periodo, df2025_periodo]:
            df_["Distrito"] = df_["Distrito"].str.strip().str.title()

        df_comparacao = pd.concat(
            [df2023_periodo, df2025_periodo],
            ignore_index=True
        )

        fig2, axes = plt.subplots(2, 3, figsize=(15, 8))
        axes = axes.flatten()

        for i, p in enumerate(poluentesclean):
            sns.boxplot(
                data=df_comparacao,
                x="Ano",
                y=p,
                ax=axes[i],
                palette="Set2"
            )
            axes[i].set_title(p)

        for j in range(len(poluentesclean), len(axes)):
            fig2.delaxes(axes[j])

        plt.tight_layout()
        st.pyplot(fig2)

# =========================================================
# ABA 3 — VARIAÇÃO PERCENTUAL
# =========================================================
with tab3:
    if df2023_clean is None or df2023_clean.empty:
        st.warning("Dataset de 2023 não disponível.")
    else:
        st.subheader("Variação Percentual por Distrito")

        df23_avg = df2023_periodo.groupby("Distrito")[poluentesclean].mean().reset_index()
        df25_avg = df2025_periodo.groupby("Distrito")[poluentesclean].mean().reset_index()

        comparacao = pd.merge(
            df23_avg, df25_avg,
            on="Distrito",
            suffixes=("_2023", "_2025")
        )

        fig3, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for i, poluente in enumerate(poluentesclean):
            comparacao[f"{poluente}_var_percent"] = (
                (comparacao[f"{poluente}_2025"] - comparacao[f"{poluente}_2023"])
                / comparacao[f"{poluente}_2023"]
            ) * 100

            comp_sorted = comparacao.sort_values(
                f"{poluente}_var_percent",
                ascending=False
            )

            axes[i].bar(
                comp_sorted["Distrito"],
                comp_sorted[f"{poluente}_var_percent"]
            )
            axes[i].axhline(0, color="gray", linestyle="--")
            axes[i].set_title(f"{poluente} (2023 → 2025)")
            axes[i].tick_params(axis="x", rotation=45)

        for j in range(len(poluentesclean), len(axes)):
            fig3.delaxes(axes[j])

        plt.tight_layout()
        st.pyplot(fig3)

# =========================================================
# ABA 4 — METEOROLOGIA VS QUALIDADE DO AR
# =========================================================
with tab4:
    st.subheader("Correlações: Meteorologia vs Qualidade do Ar (2025)")

    if df_meteo is None or df_meteo.empty:
        st.warning("Dataset meteorológico não disponível.")
    else:
        df_ar_m = df_ar.copy()
        df_ar_m["Dia"] = df_ar_m["Data"].dt.date

        df_meteo["Dia"] = pd.to_datetime(df_meteo["date"]).dt.date

        df_combinadometeoqualar = pd.merge(
            df_ar_m,
            df_meteo,
            left_on=["Distrito", "Dia"],
            right_on=["distrito", "Dia"],
            how="left"
        )

        cols_meteo = [
            "temperature_2m",
            "relative_humidity_2m",
            "rain",
            "wind_speed_80m",
            "Media_Classe"
        ]

        cols_exist = [c for c in cols_meteo if c in df_combinadometeoqualar.columns]

        if len(cols_exist) > 1:
            df_corr_input = df_combinadometeoqualar[cols_exist].dropna()

            if not df_corr_input.empty:
                fig5 = plt.figure(figsize=(10, 8))
                sns.heatmap(
                    df_corr_input.corr(),
                    annot=True,
                    cmap="coolwarm",
                    fmt=".2f",
                    vmin=-1,
                    vmax=1
                )
                st.pyplot(fig5)
            else:
                st.warning("Sem dados suficientes após limpeza.")
        else:
            st.warning("Colunas meteorológicas insuficientes após o merge.")
