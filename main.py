import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Bibliotecas de ML
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from lightgbm import LGBMRegressor

# =============================
# CONFIGURAÇÃO DA PÁGINA
# =============================
st.set_page_config(page_title="Qualidade do Ar em Portugal", layout="wide")
st.title("🌍 Qualidade do Ar e Meteorologia em Portugal")

# =============================
# PROCESSAMENTO DE DADOS (CACHE)
# =============================
@st.cache_data
def processar_dados_notebook():
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

        for p in intervalos:
            df_mediaar[f'{p}_classe'] = df_mediaar[p].apply(lambda x: classificar_proximo(x, intervalos[p]))
            
        df_ar = df_mediaar.drop(['C6H6', 'CO'], axis=1, errors='ignore')
        distritos_desejados = ['Aveiro', 'Lisboa', 'Açores', 'Setúbal', 'Leiria', 'Madeira', 'Santarém']
        df_ar = df_ar[df_ar['Distrito'].isin(distritos_desejados)].copy()
        
        colunas_classes = [c for c in df_ar.columns if c.endswith('_classe')]
        df_ar['Media_Classe'] = df_ar[colunas_classes].mean(axis=1)
        
    except Exception as e:
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

        df_sem = dfqualidadear2023.groupby(['Distrito', 'Ano', 'Semana'])[poluentes].mean(numeric_only=True).reset_index()
        dfqualidadear2023 = dfqualidadear2023.merge(df_sem, on=['Distrito', 'Ano', 'Semana'], suffixes=('', '_media'), how='left')
        for p in poluentes:
            if f'{p}_media' in dfqualidadear2023.columns:
                dfqualidadear2023[p] = dfqualidadear2023[p].fillna(dfqualidadear2023[f'{p}_media'])
        
        df2023_media = dfqualidadear2023.groupby(["Distrito", "Data-Hora","Semana","Ano"])[poluentes].mean(numeric_only=True).reset_index()
        
        for p in intervalos:
            df2023_media[f'{p}_classe'] = df2023_media[p].apply(lambda x: classificar_proximo(x, intervalos[p]))
        
        cls_cols = [c for c in df2023_media.columns if c.endswith('_classe')]
        df2023_media['Media_Classe'] = df2023_media[cls_cols].mean(axis=1)
        
        df2023_mediaclean = df2023_media.drop(['C6H6', 'CO'], axis=1, errors='ignore')
        df2023_mediaclean = df2023_mediaclean[df2023_mediaclean['Distrito'].isin(distritos_desejados)].copy()
        df2023_mediaclean['Data'] = pd.to_datetime(df2023_mediaclean['Data-Hora']).dt.date
        df2023_mediaclean['Data'] = pd.to_datetime(df2023_mediaclean['Data'])
        
    except:
        df2023_mediaclean = None

    # --- C. METEOROLOGIA ---
    try:
        df_meteo = pd.read_csv("dataset_meteorologico_portugal.csv")
        # CORREÇÃO CRÍTICA PARA ERRO .DT E TIMEZONE
        df_meteo["date"] = pd.to_datetime(df_meteo["date"], utc=True)
        df_meteo["date"] = df_meteo["date"].dt.tz_localize(None)
        
        if 'distrito' in df_meteo.columns:
            df_meteo["distrito"] = df_meteo["distrito"].astype(str).str.strip().str.title()
            
    except:
        df_meteo = None

    return df_ar, df2023_mediaclean, df_meteo, distritos_desejados

# Carregar dados
df_ar, df2023_clean, df_meteo, distritos_desejados = processar_dados_notebook()

if df_ar is None:
    st.error("Erro ao carregar dados de 2025. Verifique os ficheiros.")
    st.stop()

# ==============================================================================
# SIDEBAR - NAVEGAÇÃO
# ==============================================================================
st.sidebar.title("📌 Navegação")
section = st.sidebar.radio(
    "Escolha a secção",
    ["Datasets", "EDA", "Machine Learning", "SVR Autoregressivo"]
)

poluentesclean = ['NO2', 'O3', 'PM2.5', 'PM10', 'SO2']

# ==============================================================================
# 1. DATASETS
# ==============================================================================
if section == "Datasets":
    st.header("📂 Datasets Utilizados")
    if df_meteo is not None:
        st.subheader("Meteorologia")
        st.dataframe(df_meteo.head())
    
    st.subheader("Qualidade do Ar 2025 (Processado)")
    st.dataframe(df_ar.head())

    if df2023_clean is not None:
        st.subheader("Qualidade do Ar 2023 (Processado)")
        st.dataframe(df2023_clean.head())

# ==============================================================================
# 2. EDA
# ==============================================================================
elif section == "EDA":
    st.header("Análise Exploratória")

    st.subheader("1. Componentes Diários 2025")
    df_por_dia = df_ar.groupby("Data")[poluentesclean].mean().reset_index()
    fig1, axes = plt.subplots(nrows=1, ncols=len(poluentesclean), figsize=(18, 5))
    for i, col in enumerate(poluentesclean):
        sns.boxplot(y=df_por_dia[col], ax=axes[i], color="skyblue")
        axes[i].set_title(f'{col} 2025')
        axes[i].set_ylabel("")
    st.pyplot(fig1)

    if df2023_clean is not None:
        st.subheader("2. Comparação 2023 vs 2025")
        datas_2025 = df_ar["Data"].dt.strftime("%m-%d").unique()
        df23_p = df2023_clean[df2023_clean["Data"].dt.strftime("%m-%d").isin(datas_2025)].copy()
        
        df23_p["Ano"] = 2023
        df25_p = df_ar.copy()
        df25_p["Ano"] = 2025
        df23_p["Distrito"] = df23_p["Distrito"].str.strip().str.title()
        df25_p["Distrito"] = df25_p["Distrito"].str.strip().str.title()
        
        df_comp = pd.concat([df23_p, df25_p], ignore_index=True)
        
        fig2, axes = plt.subplots(2, 3, figsize=(15,8))
        axes = axes.flatten()
        for i, p in enumerate(poluentesclean):
            sns.boxplot(data=df_comp, x='Ano', y=p, ax=axes[i], palette="Set2")
            axes[i].set_title(p)
        for j in range(len(poluentesclean), len(axes)): fig2.delaxes(axes[j])
        st.pyplot(fig2)
        
        st.subheader("3. Variação Percentual por Distrito")
        df23_avg = df23_p.groupby("Distrito")[poluentesclean].mean().reset_index()
        df25_avg = df25_p.groupby("Distrito")[poluentesclean].mean().reset_index()
        comp = pd.merge(df23_avg, df25_avg, on="Distrito", suffixes=("_2023", "_2025"))
        
        fig3, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
        axes = axes.flatten()
        for i, p in enumerate(poluentesclean):
            comp[f"{p}_var"] = ((comp[f"{p}_2025"] - comp[f"{p}_2023"]) / comp[f"{p}_2023"]) * 100
            comp_s = comp.sort_values(f"{p}_var", ascending=False)
            axes[i].bar(comp_s["Distrito"], comp_s[f"{p}_var"])
            axes[i].axhline(0, color="gray", linestyle="--")
            axes[i].set_title(f"{p} Var %")
            axes[i].tick_params(axis='x', rotation=45)
        for j in range(len(poluentesclean), len(axes)): fig3.delaxes(axes[j])
        st.pyplot(fig3)
        
        st.subheader("4. Média Geral da Classe")
        m_df = pd.DataFrame({
            'Ano': ['2023', '2025'],
            'Media': [df23_p['Media_Classe'].mean(), df25_p['Media_Classe'].mean()]
        })
        fig4 = plt.figure(figsize=(6,5))
        ax = sns.barplot(data=m_df, x='Ano', y='Media', palette=['#FFA500', '#1F77B4'])
        for p in ax.patches:
            ax.annotate(f'{p.get_height():.2f}', (p.get_x()+p.get_width()/2., p.get_height()), ha='center', va='bottom')
        st.pyplot(fig4)

    # Correlações
    if df_meteo is not None:
        st.subheader("5. Correlações (Meteo vs Ar)")
        df_ar_m = df_ar.copy()
        df_ar_m["Dia"] = df_ar_m["Data"].dt.date
        df_meteo["Dia"] = df_meteo["date"].dt.date
        
        df_merged = pd.merge(df_ar_m, df_meteo, left_on=["Distrito", "Dia"], right_on=["distrito", "Dia"], how="left")
        
        cols_m = ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_80m", "Media_Classe"]
        valid_cols = [c for c in cols_m if c in df_merged.columns]
        
        if len(valid_cols) > 1:
            fig5 = plt.figure(figsize=(10, 8))
            sns.heatmap(df_merged[valid_cols].dropna().corr(), annot=True, cmap="coolwarm", fmt=".2f")
            st.pyplot(fig5)

# ==============================================================================
# 3. MACHINE LEARNING (IGUAL AO SNIPPET + VISUALIZAÇÃO dfL)
# ==============================================================================
elif section == "Machine Learning":
    st.header("🤖 Machine Learning (Lisboa)")
    st.markdown("Resultados da Regressão com GridSearch (sem Lags, apenas Meteo base).")
    
    if df_meteo is None:
        st.error("Sem dados de meteorologia.")
        st.stop()
        
    # --- PREPARAÇÃO DOS DADOS IGUAL AO NOTEBOOK ---
    df_ar_ml = df_ar.rename(columns={'Data': 'date', 'Distrito': 'distrito'})
    distritos_validos = df_ar_ml['distrito'].unique()
    df_meteo_filtrado = df_meteo[df_meteo['distrito'].isin(distritos_validos)]
    df_meteo_filtrado['date'] = df_meteo_filtrado['date'].dt.floor('D')
    df_meteo_filtrado = df_meteo_filtrado.groupby(['date', 'distrito']).mean(numeric_only=True).reset_index()
    
    df_merged = pd.merge(df_ar_ml, df_meteo_filtrado, on=['date', 'distrito'], how='inner')
    df_Model = df_merged.dropna().copy()
    
    # Filtro Lisboa
    dfL = df_Model[df_Model["distrito"] == "Lisboa"].copy()
    dfL = dfL.sort_values("date").reset_index(drop=True)
    
    if dfL.empty:
        st.warning("Sem dados combinados para Lisboa.")
    else:
        # VISUALIZAÇÃO PEDIDA DO DATAFRAME
        st.subheader("Dataframe usado para ML (dfL)")
        st.dataframe(dfL)

        # Features EXATAS do teu loop (Só meteo, SEM LAGS AQUI)
        X_cols = [
            "rain", "temperature_2m", "relative_humidity_2m",
            "temperature_80m", "wind_speed_80m", "wind_direction_80m",
            "temperature_2m_max", "temperature_2m_min"
        ]
        Y_cols = ["O3", "NO2", "SO2", "PM10", "PM2.5"]
        
        if st.button("Treinar Modelos (GridSearch)"):
            results = []
            
            # PARÂMETROS IGUAIS AO NOTEBOOK
            param_grids = {
                "Random


