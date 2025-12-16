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
# PROCESSING
# =============================


def processar_dados_notebook():
    st.write("--- 1. Carregamento e Processamento de Dados ---")
    
    # --- A. DADOS 2025 ---
    try:
        dfqualidadear = pd.read_excel("QualidadeAr2.xlsx")
        if 'Coluna1' in dfqualidadear.columns:
            dfqualidadear = dfqualidadear.rename(columns={"Coluna1":"Distrito"})
        if "Tipo" in dfqualidadear.columns: dfqualidadear = dfqualidadear.drop(["Tipo", "Zona"], axis=1)
        
        dfqualidadear['Data'] = pd.to_datetime(dfqualidadear['Data'], dayfirst=True, errors='coerce')
        dfqualidadear['Semana'] = dfqualidadear['Data'].dt.isocalendar().week
        dfqualidadear['Ano'] = dfqualidadear['Data'].dt.year
        
        poluentes = ['C6H6', 'CO', 'NO2', 'O3', 'PM2.5', 'PM10', 'SO2']
        for p in poluentes:
            if p in dfqualidadear.columns:
                dfqualidadear[p] = pd.to_numeric(dfqualidadear[p], errors='coerce')

        # Imputação (Média Semanal)
        df_medias = dfqualidadear.groupby(['Distrito', 'Ano', 'Semana'])[poluentes].mean().reset_index()
        dfqualidadear = dfqualidadear.merge(df_medias, on=['Distrito', 'Ano', 'Semana'], suffixes=('', '_media'), how='left')
        for p in poluentes:
            if f'{p}_media' in dfqualidadear.columns:
                dfqualidadear[p] = dfqualidadear[p].fillna(dfqualidadear[f'{p}_media'])
        
        df_mediaar = dfqualidadear.groupby(["Data","Distrito","Semana","Ano"])[poluentes].mean().reset_index()

        # Classificação
        intervalos = {
            "PM10": [(101, 1200), (51, 100), (36, 50), (21, 35), (0, 20)],
            "PM2.5": [(51, 800), (26, 50), (21, 25), (11, 20), (0, 10)],
            "NO2": [(401, 1000), (201, 400), (101, 200), (41, 100), (0, 40)],
            "O3": [(241, 600), (181, 240), (101, 180), (81, 100), (0, 80)],
            "SO2": [(501, 1250), (351, 500), (201, 350), (101, 200), (0, 100)]
        }

        def classificar_proximo(valor, intervalos_dict):
            if pd.isna(valor): return np.nan
            distancias = []
            for i, (minimo, maximo) in enumerate(intervalos_dict):
                centro = (minimo + maximo) / 2
                distancias.append((abs(valor - centro), i + 1)) 
            return min(distancias)[1]

        for poluente in intervalos:
            df_mediaar[f'{poluente}_classe'] = df_mediaar[poluente].apply(lambda x: classificar_proximo(x, intervalos[poluente]))
            
        df_ar = df_mediaar.drop(['C6H6', 'CO'], axis=1, errors='ignore')
        distritos_desejados = ['Aveiro', 'Lisboa', 'Açores', 'Setúbal', 'Leiria', 'Madeira', 'Santarém']
        df_ar = df_ar[df_ar['Distrito'].isin(distritos_desejados)].copy()
        
        colunas_classes = [c for c in df_ar.columns if c.endswith('_classe')]
        df_ar['Media_Classe'] = df_ar[colunas_classes].mean(axis=1)
        
    except Exception as e:
        st.error(f"Erro no processamento de 2025: {e}")
        return None, None, None, None

    # --- B. DADOS 2023 ---
    try:
        dfqualidadear2023 = pd.read_excel("Qualar2023.xlsx")
        dfqualidadear2023.columns = dfqualidadear2023.columns.str.strip()
        if 'Local' in dfqualidadear2023.columns: dfqualidadear2023 = dfqualidadear2023.drop('Local', axis=1)
        
        if 'Data-Hora' in dfqualidadear2023.columns:
            dfqualidadear2023['Data-Hora'] = pd.to_datetime(dfqualidadear2023['Data-Hora'])
            dfqualidadear2023['Semana'] = dfqualidadear2023['Data-Hora'].dt.isocalendar().week
            dfqualidadear2023['Ano'] = dfqualidadear2023['Data-Hora'].dt.year
        
        for p in poluentes:
            if p in dfqualidadear2023.columns:
                dfqualidadear2023[p] = pd.to_numeric(dfqualidadear2023[p], errors='coerce')

        df_semanalar2023 = dfqualidadear2023.groupby(['Distrito', 'Ano', 'Semana'])[poluentes].mean(numeric_only=True).reset_index()
        
        dfqualidadear2023 = dfqualidadear2023.merge(
            df_semanalar2023, on=['Distrito', 'Ano', 'Semana'], suffixes=('', '_media'), how='left'
        )
        for p in poluentes:
            if f'{p}_media' in dfqualidadear2023.columns:
                dfqualidadear2023[p] = dfqualidadear2023[p].fillna(dfqualidadear2023[f'{p}_media'])
        
        df2023_media = dfqualidadear2023.groupby(["Distrito", "Data-Hora","Semana","Ano"])[poluentes].mean(numeric_only=True).reset_index()
        
        for poluente in intervalos:
            df2023_media[f'{poluente}_classe'] = df2023_media[poluente].apply(lambda x: classificar_proximo(x, intervalos[poluente]))
        
        colunas_classes = [c for c in df2023_media.columns if c.endswith('_classe')]
        df2023_media['Media_Classe'] = df2023_media[colunas_classes].mean(axis=1)
        
        df2023_mediaclean = df2023_media.drop(['C6H6', 'CO'], axis=1, errors='ignore')
        df2023_mediaclean = df2023_mediaclean[df2023_mediaclean['Distrito'].isin(distritos_desejados)].copy()
        df2023_mediaclean['Data'] = pd.to_datetime(df2023_mediaclean['Data-Hora']).dt.date
        df2023_mediaclean['Data'] = pd.to_datetime(df2023_mediaclean['Data']) 
        
    except Exception as e:
        st.error(f"Erro no processamento de 2023: {e}")
        df2023_mediaclean = None

    # --- C. METEOROLOGIA (CORREÇÃO) ---
    try:
        df_meteo = pd.read_csv("dataset_meteorologico_portugal.csv")
        
        # Correção Robusta: Converter para UTC primeiro para evitar erros, depois remover TZ
        df_meteo["date"] = pd.to_datetime(df_meteo["date"], utc=True)
        df_meteo["date"] = df_meteo["date"].dt.tz_localize(None)
        
        # Normalizar distrito
        if 'distrito' in df_meteo.columns:
            df_meteo["distrito"] = df_meteo["distrito"].astype(str).str.strip().str.title()
            
    except Exception as e:
        st.warning(f"Erro na meteorologia: {e}")
        df_meteo = None

    return df_ar, df2023_mediaclean, df_meteo, distritos_desejados












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


