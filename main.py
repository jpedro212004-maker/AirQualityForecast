import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import os
import time

# Bibliotecas de ML (Exatamente como no teu notebook)
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMRegressor

# Configuração da Página
st.set_page_config(page_title="Replica Notebook", layout="wide")
st.title("Notebook: Análise e Modelação da Qualidade do Ar")

# ==============================================================================
# 1. FUNÇÕES DE PROCESSAMENTO (Lógica exata do Notebook)
# ==============================================================================

def processar_dados_notebook():
    """
    Executa o pipeline de tratamento de dados EXATAMENTE como no notebook:
    1. Leitura
    2. Conversão de Tipos
    3. Imputação (Fillna com médias semanais)
    4. Criação de Classes
    5. Filtragem de Datas e Distritos
    """
    st.write("--- 1. Carregamento e Processamento de Dados ---")
    
    # --------------------------------------------------------------------------
    # A. DADOS 2025
    # --------------------------------------------------------------------------
    try:
        dfqualidadear = pd.read_excel("QualidadeAr2.xlsx")
        # Renomear (Notebook code)
        if 'Coluna1' in dfqualidadear.columns:
            dfqualidadear = dfqualidadear.rename(columns={"Coluna1":"Distrito"})
        if "Tipo" in dfqualidadear.columns: dfqualidadear = dfqualidadear.drop(["Tipo", "Zona"], axis=1)
        
        # Converter Data
        dfqualidadear['Data'] = pd.to_datetime(dfqualidadear['Data'], dayfirst=True, errors='coerce')
        
        # Extrair semana e ano
        dfqualidadear['Semana'] = dfqualidadear['Data'].dt.isocalendar().week
        dfqualidadear['Ano'] = dfqualidadear['Data'].dt.year
        
        # Converter numéricos
        poluentes = ['C6H6', 'CO', 'NO2', 'O3', 'PM2.5', 'PM10', 'SO2']
        for p in poluentes:
            if p in dfqualidadear.columns:
                dfqualidadear[p] = pd.to_numeric(dfqualidadear[p], errors='coerce')

        # --- IMPUTAÇÃO (Passo Crucial do Notebook) ---
        # Calcular médias semanais por distrito
        df_medias = dfqualidadear.groupby(['Distrito', 'Ano', 'Semana'])[poluentes].mean().reset_index()
        
        # Juntar ao DataFrame original
        dfqualidadear = dfqualidadear.merge(df_medias, on=['Distrito', 'Ano', 'Semana'], suffixes=('', '_media'), how='left')
        
        # Preencher NaNs com as médias semanais
        for p in poluentes:
            if f'{p}_media' in dfqualidadear.columns:
                dfqualidadear[p] = dfqualidadear[p].fillna(dfqualidadear[f'{p}_media'])
        
        # Calcular Média Agrupada (df_mediaar)
        df_mediaar = dfqualidadear.groupby(["Data","Distrito","Semana","Ano"])[poluentes].mean().reset_index()

        # --- CLASSIFICAÇÃO ---
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
            
        # Limpeza final e Filtro Distritos (Notebook Code)
        df_ar = df_mediaar.drop(['C6H6', 'CO'], axis=1, errors='ignore')
        distritos_desejados = ['Aveiro', 'Lisboa', 'Açores', 'Setúbal', 'Leiria', 'Madeira', 'Santarém']
        df_ar = df_ar[df_ar['Distrito'].isin(distritos_desejados)].copy()
        
        # Média das classes
        colunas_classes = [c for c in df_ar.columns if c.endswith('_classe')]
        df_ar['Media_Classe'] = df_ar[colunas_classes].mean(axis=1)
        
    except Exception as e:
        st.error(f"Erro no processamento de 2025: {e}")
        return None, None, None, None

    # --------------------------------------------------------------------------
    # B. DADOS 2023
    # --------------------------------------------------------------------------
    try:
        dfqualidadear2023 = pd.read_excel("Qualar2023.xlsx")
        dfqualidadear2023.columns = dfqualidadear2023.columns.str.strip()
        if 'Local' in dfqualidadear2023.columns: dfqualidadear2023 = dfqualidadear2023.drop('Local', axis=1)
        
        # Tratamento de Datas
        if 'Data-Hora' in dfqualidadear2023.columns:
            dfqualidadear2023['Data-Hora'] = pd.to_datetime(dfqualidadear2023['Data-Hora'])
            dfqualidadear2023['Semana'] = dfqualidadear2023['Data-Hora'].dt.isocalendar().week
            dfqualidadear2023['Ano'] = dfqualidadear2023['Data-Hora'].dt.year
        
        # Imputação (Média Semanal) - Replica exata
        poluentes = ['C6H6', 'CO', 'NO2', 'O3', 'PM2.5', 'PM10', 'SO2']
        # Forçar numérico primeiro
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
        
        # Agrupar médias diárias
        df2023_media = dfqualidadear2023.groupby(["Distrito", "Data-Hora","Semana","Ano"])[poluentes].mean(numeric_only=True).reset_index()
        
        # Classificação
        for poluente in intervalos:
            df2023_media[f'{poluente}_classe'] = df2023_media[poluente].apply(lambda x: classificar_proximo(x, intervalos[poluente]))
        
        colunas_classes = [c for c in df2023_media.columns if c.endswith('_classe')]
        df2023_media['Media_Classe'] = df2023_media[colunas_classes].mean(axis=1)
        
        # Limpeza final 2023
        df2023_mediaclean = df2023_media.drop(['C6H6', 'CO'], axis=1, errors='ignore')
        df2023_mediaclean = df2023_mediaclean[df2023_mediaclean['Distrito'].isin(distritos_desejados)].copy()
        
        df2023_mediaclean['Data'] = pd.to_datetime(df2023_mediaclean['Data-Hora']).dt.date
        df2023_mediaclean['Data'] = pd.to_datetime(df2023_mediaclean['Data']) # Garantir datetime64
        
    except Exception as e:
        st.error(f"Erro no processamento de 2023: {e}")
        return df_ar, None, None, None

    # --------------------------------------------------------------------------
    # C. METEOROLOGIA
    # --------------------------------------------------------------------------
    try:
        df_meteo = pd.read_csv("dataset_meteorologico_portugal.csv")
        df_meteo["date"] = pd.to_datetime(df_meteo["date"], errors='coerce')
        # Remover Timezone para bater com Excel
        df_meteo["date"] = df_meteo["date"].dt.tz_localize(None) 
    except Exception as e:
        st.warning(f"Erro na meteo: {e}")
        df_meteo = None

    return df_ar, df2023_mediaclean, df_meteo, distritos_desejados

# ==============================================================================
# EXECUÇÃO DA LÓGICA
# ==============================================================================

# Executar a função gigante de tratamento
df_ar, df2023_clean, df_meteo, distritos_desejados = processar_dados_notebook()

if df_ar is None:
    st.stop()

poluentesclean = ['NO2', 'O3', 'PM2.5', 'PM10', 'SO2']

# ==============================================================================
# 2. EDA (GRÁFICOS)
# ==============================================================================
st.write("--- 2. Análise Exploratória (EDA) ---")

# A. Boxplots 2025
st.subheader("Componentes Diários 2025")
df_por_dia = df_ar.groupby("Data")[poluentesclean].mean().reset_index()

fig1, axes = plt.subplots(nrows=1, ncols=len(poluentesclean), figsize=(18, 5))
for i, col in enumerate(poluentesclean):
    sns.boxplot(y=df_por_dia[col], ax=axes[i], color="skyblue")
    axes[i].set_title(f'Boxplot de {col} em 2025')
    axes[i].set_ylabel("")
plt.tight_layout()
st.pyplot(fig1)

# B. Comparação 2023 vs 2025 (Com filtro de data exato do notebook)
if df2023_clean is not None:
    st.subheader("Comparação 2023 vs 2025")
    
    # Filtro de datas para igualar o notebook
    datas_2025 = df_ar["Data"].dt.strftime("%m-%d").unique()
    df2023_periodo = df2023_clean[
        df2023_clean["Data"].dt.strftime("%m-%d").isin(datas_2025)
    ].copy()
    
    # Preparar df conjunto
    df2023_periodo["Ano"] = 2023
    df2025_periodo = df_ar.copy()
    df2025_periodo["Ano"] = 2025
    
    # Uniformizar nomes de distritos
    df2023_periodo["Distrito"] = df2023_periodo["Distrito"].str.strip().str.title()
    df2025_periodo["Distrito"] = df2025_periodo["Distrito"].str.strip().str.title()
    
    df_comparacao = pd.concat([df2023_periodo, df2025_periodo], ignore_index=True)
    
    # Boxplots Comparativos
    fig2, axes = plt.subplots(2, 3, figsize=(15,8))
    axes = axes.flatten()
    for i, p in enumerate(poluentesclean):
        sns.boxplot(data=df_comparacao, x='Ano', y=p, ax=axes[i], palette="Set2")
        axes[i].set_title(p)
    for j in range(len(poluentesclean), len(axes)):
        fig2.delaxes(axes[j])
    plt.tight_layout()
    st.pyplot(fig2)
    
    # Variação Percentual
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
    
    # Média Geral da Classe
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

# C. Correlações com Meteorologia
if df_meteo is not None:
    st.subheader("Correlações: Meteorologia vs Qualidade do Ar (2025)")
    
    # Preparar Merge
    df_ar_m = df_ar.copy()
    df_ar_m["Dia"] = df_ar_m["Data"].dt.date
    df_meteo["Dia"] = df_meteo["date"].dt.date
    df_meteo["distrito"] = df_meteo["distrito"].str.strip().str.title()
    
    df_combinadometeoqualar = pd.merge(
        df_ar_m, df_meteo, left_on=["Distrito", "Dia"], right_on=["distrito", "Dia"], how="left"
    )
    
    cols_meteo = ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_80m", "Media_Classe"]
    df_corr = df_combinadometeoqualar[cols_meteo].dropna()
    
    fig5 = plt.figure(figsize=(10, 8))
    sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    st.pyplot(fig5)
    
    st.subheader("Correlações Detalhadas (Componentes)")
    colunas_meteo = ["temperature_2m", "relative_humidity_2m", "rain", "temperature_80m", "wind_speed_80m", "wind_direction_80m", "temperature_2m_max", "temperature_2m_min", "uv_index_max"]
    colunas_disp = [c for c in colunas_meteo + poluentesclean if c in df_combinadometeoqualar.columns]
    
    corr_full = df_combinadometeoqualar[colunas_disp].corr()
    corr_sub = corr_full.loc[[c for c in colunas_meteo if c in corr_full.index], [c for c in poluentesclean if c in corr_full.columns]]
    
    fig6 = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_sub, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    st.pyplot(fig6)

# ==============================================================================
# 3. MACHINE LEARNING
# ==============================================================================
st.write("--- 3. Machine Learning ---")

# Preparar dados para ML (Merge feito anteriormente)
df_Model = df_combinadometeoqualar.dropna().copy()
# Filtrar Lisboa (Notebook logic)
dfL = df_Model[df_Model["distrito"] == "Lisboa"].copy()
dfL["date"] = pd.to_datetime(dfL["date"])
dfL = dfL.sort_values("date").reset_index(drop=True)

# Feature Engineering do Notebook
X_cols_base = ["rain", "temperature_2m", "relative_humidity_2m", "temperature_80m", "wind_speed_80m", "wind_direction_80m", "temperature_2m_max", "temperature_2m_min"]
Y_cols = ["O3", "NO2", "SO2", "PM10", "PM2.5"]

for col in Y_cols:
    dfL[f"{col}_lag1"] = dfL[col].shift(1)
    dfL[f"{col}_lag2"] = dfL[col].shift(2)
    dfL[f"{col}_roll3"] = dfL[col].rolling(3).mean()

dfL["month"] = dfL["date"].dt.month
dfL["weekday"] = dfL["date"].dt.weekday

X_cols_final = X_cols_base + [f"{col}_lag1" for col in Y_cols] + [f"{col}_lag2" for col in Y_cols] + [f"{col}_roll3" for col in Y_cols] + ["month", "weekday"]

# ------------------------------------------------------------------------------
# PARTE INTENSIVA: TREINO DE MODELOS (WRAPPER)
# ------------------------------------------------------------------------------
st.markdown("### Treino de Modelos para Lisboa")
st.info("O treino dos modelos (GridSearch) é computacionalmente pesado. Clique no botão abaixo para executar.")

if st.button("Executar Treino de Modelos (Pode demorar)"):
    
    # 1. Regressão (RandomForest, LightGBM, MLP)
    results = []
    
    # Modelos (Versão um pouco mais leve do Grid para não crashar a Cloud)
    # Reduzi o cv=3 e n_estimators para garantir execução
    param_grids = {
        "RandomForest": {"n_estimators": [50], "max_depth": [5, 10]}, 
        "LightGBM": {"n_estimators": [50], "learning_rate": [0.1]},
        "MLP": {"hidden_layer_sizes": [(32,)], "max_iter": [200]}
    }
    
    models_dict = {
        "RandomForest": RandomForestRegressor(random_state=42),
        "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
        "MLP": MLPRegressor(random_state=42)
    }
    
    progress_bar = st.progress(0)
    total_steps = len(Y_cols)
    
    for i, target in enumerate(Y_cols):
        y = dfL[target].dropna()
        # Alinhar X com y (devido aos lags que geram NaNs)
        X = dfL[X_cols_final].loc[y.index]
        
        # Imputer simples
        imputer = SimpleImputer(strategy="mean")
        X_imp = imputer.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, shuffle=False)
        
        for name, model in models_dict.items():
            grid = GridSearchCV(model, param_grids[name], cv=2, scoring="neg_mean_absolute_error", n_jobs=1)
            grid.fit(X_train, y_train)
            
            y_pred = grid.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results.append({"Poluente": target, "Modelo": name, "MAE": mae, "R2": r2})
        
        progress_bar.progress((i + 1) / total_steps)
            
    st.write("Resultados Regressão:")
    st.dataframe(pd.DataFrame(results))

    # 2. SVR Autoregressivo (Previsão Media_Classe)
    st.markdown("### SVR Autoregressivo (Media_Classe)")
    
    df_class = dfL.copy()
    # Criar lags na Media_Classe
    for lag in range(1, 8):
        df_class[f"lag{lag}"] = df_class["Media_Classe"].shift(lag)
    
    df_class = df_class.dropna()
    X_svr = df_class[[f"lag{i}" for i in range(1, 8)]]
    y_svr = df_class["Media_Classe"]
    
    # Treino
    model_ar = SVR(C=10, epsilon=0.1, gamma=0.01)
    model_ar.fit(X_svr, y_svr)
    y_pred_in = model_ar.predict(X_svr)
    
    # Métricas SVR
    c1, c2, c3 = st.columns(3)
    c1.metric("MAE", f"{mean_absolute_error(y_svr, y_pred_in):.4f}")
    c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_svr, y_pred_in)):.4f}")
    c3.metric("R2", f"{r2_score(y_svr, y_pred_in):.4f}")
    
    # Gráfico SVR
    fig_svr = go.Figure()
    fig_svr.add_trace(go.Scatter(x=df_class["date"], y=y_svr, mode="lines", name="Real", line=dict(color="blue")))
    fig_svr.add_trace(go.Scatter(x=df_class["date"], y=y_pred_in, mode="lines", name="Previsto (SVR)", line=dict(color="red")))
    fig_svr.update_layout(title="SVR Autoregressivo - Real vs Previsto", xaxis_title="Data", template="plotly_white")
    st.plotly_chart(fig_svr, use_container_width=True)

else:
    st.warning("⚠️ Atenção: Executar o GridSearch em tempo real consome muita memória. Execute apenas se necessário.")

