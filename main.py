import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

# Bibliotecas de ML
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.svm import SVR, SVC
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from lightgbm import LGBMRegressor

# =============================
# CONFIG
# =============================
st.set_page_config(page_title="Qualidade do Ar em Portugal", layout="wide")
st.title("Qualidade do Ar em Portugal")

# =============================
# PROCESSING (CACHE)
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
        # NORMALIZAÇÃO DE DATA
        dfqualidadear['Data'] = dfqualidadear['Data'].dt.normalize()
        
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
        df_meteo["date"] = pd.to_datetime(df_meteo["date"], utc=True)
        df_meteo["date"] = df_meteo["date"].dt.tz_localize(None)
        # NORMALIZAÇÃO DE DATA
        df_meteo["date"] = df_meteo["date"].dt.normalize()
        
        if 'distrito' in df_meteo.columns:
            df_meteo["distrito"] = df_meteo["distrito"].astype(str).str.strip().str.title()
            
    except:
        df_meteo = None

    return df_ar, df2023_mediaclean, df_meteo, distritos_desejados

# Carregar dados globalmente
df_ar, df2023_clean, df_meteo, distritos_desejados = processar_dados_notebook()

if df_ar is None:
    st.error("Erro ao carregar dados de 2025. Verifique os ficheiros.")
    st.stop()

# ==============================================================================
# SIDEBAR
# ==============================================================================
st.sidebar.title("Navegação")
section = st.sidebar.radio(
    "Escolha a secção",
    ["Sobre o Trabalho", "Datasets", "EDA", "Machine Learning (Base)", "Machine Learning (Avançado)", "Classificação (Prever Classes)", "SVR","Conclusão"])

poluentesclean = ['NO2', 'O3', 'PM2.5', 'PM10', 'SO2']

# ==============================================================================
# 0. SOBRE O TRABALHO 
# ==============================================================================
# ==============================================================================
# 0. SOBRE O TRABALHO
# ==============================================================================
if section == "Sobre o Trabalho":
    st.header("Sobre o Trabalho")
    st.markdown("""
    ###
    
    Este projeto tem como principal objetivo analisar a qualidade do ar em Portugal, focando-se em dois pontos essenciais:

    1.  **Entender as Relações:**
        Queremos perceber de que forma a qualidade do ar é influenciada por outros fatores externos, tais como:
        * **Condições Meteorológicas** (temperatura, chuva, vento).
        * **Ocorrência de Incêndios**.
        * **Localização**.
        * **Fatores Populacionais e Ambientais**.

    2.  **Fazer Previsões:**
        Para além de analisar o passado, o nosso objetivo é **prever o futuro**. Utilizamos modelos de *Machine Learning* para estimar como estará a qualidade do ar nos próximos dias em um distrito de Portugal, com base no histórico e nas previsões do tempo.
    """)

 

# ==============================================================================
# 1. DATASETS
# ==============================================================================
elif section == "Datasets":
    st.header("Datasets Utilizados")
    if df_meteo is not None:
        st.subheader("Meteorologia (Normalizado)")
        st.dataframe(df_meteo)
    
    st.subheader("Qualidade do Ar 2025 (Processado)")
    st.dataframe(df_ar)

    if df2023_clean is not None:
        st.subheader("Qualidade do Ar 2023 (Processado)")
        st.dataframe(df2023_clean)

# ==============================================================================
# 2. EDA - VERSÃO COMPACTA
# ==============================================================================
elif section == "EDA":
    st.header("Análise Exploratória")

    # 1. Componentes Diários 2025 - Reduzido na altura e largura
    st.subheader("1. Poluentes 2025")
    df_por_dia = df_ar.groupby("Data")[poluentesclean].mean().reset_index()
    fig1, axes = plt.subplots(nrows=1, ncols=len(poluentesclean), figsize=(12, 3)) # Altura 3 é bem pequena
    for i, col in enumerate(poluentesclean):
        sns.boxplot(y=df_por_dia[col], ax=axes[i], color="skyblue")
        axes[i].set_title(col, fontsize=9)
        axes[i].set_ylabel("")
        axes[i].tick_params(labelsize=8)
    plt.tight_layout()
    st.pyplot(fig1)

    if df2023_clean is not None:
        st.divider()
        # Colocamos os gráficos de comparação lado a lado para não ocupar verticalmente
        c1, c2 = st.columns(2)
        
        with c1:
            st.subheader("2. 2023 vs 2025")
            datas_2025 = df_ar["Data"].dt.strftime("%m-%d").unique()
            df23_p = df2023_clean[df2023_clean["Data"].dt.strftime("%m-%d").isin(datas_2025)].copy()
            df23_p["Ano"] = 2023
            df25_p = df_ar.copy()
            df25_p["Ano"] = 2025
            df_comp = pd.concat([df23_p, df25_p], ignore_index=True)
            
            fig2, axes = plt.subplots(2, 3, figsize=(8, 5)) # Tamanho reduzido para coluna
            axes = axes.flatten()
            for i, p in enumerate(poluentesclean):
                sns.boxplot(data=df_comp, x='Ano', y=p, ax=axes[i], palette="Set2")
                axes[i].set_title(p, fontsize=8)
                axes[i].set_ylabel("")
                axes[i].tick_params(labelsize=7)
            for j in range(len(poluentesclean), len(axes)): fig2.delaxes(axes[j])
            plt.tight_layout()
            st.pyplot(fig2)

        with c2:
            st.subheader("3. Média de Classe")
            m_df = pd.DataFrame({
                'Ano': ['2023', '2025'],
                'Media': [df23_p['Media_Classe'].mean(), df25_p['Media_Classe'].mean()]
            })
            fig4 = plt.figure(figsize=(4, 4)) # Gráfico quase quadrado e pequeno
            ax = sns.barplot(data=m_df, x='Ano', y='Media', palette=['#FFA500', '#1F77B4'])
            plt.xticks(fontsize=8)
            plt.yticks(fontsize=8)
            st.pyplot(fig4)

        st.subheader("4. Variação % por Distrito")
        df23_avg = df23_p.groupby("Distrito")[poluentesclean].mean().reset_index()
        df25_avg = df25_p.groupby("Distrito")[poluentesclean].mean().reset_index()
        comp = pd.merge(df23_avg, df25_avg, on="Distrito", suffixes=("_2023", "_2025"))
        
        fig3, axes = plt.subplots(nrows=2, ncols=3, figsize=(10, 5)) # Altura reduzida de 10 para 5
        axes = axes.flatten()
        for i, p in enumerate(poluentesclean):
            comp[f"{p}_var"] = ((comp[f"{p}_2025"] - comp[f"{p}_2023"]) / comp[f"{p}_2023"]) * 100
            comp_s = comp.sort_values(f"{p}_var", ascending=False)
            axes[i].bar(comp_s["Distrito"], comp_s[f"{p}_var"])
            axes[i].set_title(f"{p} Var %", fontsize=8)
            axes[i].tick_params(axis='x', rotation=45, labelsize=7)
        for j in range(len(poluentesclean), len(axes)): fig3.delaxes(axes[j])
        plt.tight_layout()
        st.pyplot(fig3)

    if df_meteo is not None:
        st.divider()
        st.subheader("5 & 6. Heatmaps de Correlação")
        h1, h2 = st.columns(2)
        
        df_ar_m = df_ar.copy()
        df_ar_m["Dia"] = df_ar_m["Data"].dt.normalize()
        df_meteo_m = df_meteo.copy()
        df_meteo_m["Dia"] = df_meteo_m["date"].dt.normalize()
        df_merged = pd.merge(df_ar_m, df_meteo_m, left_on=["Distrito", "Dia"], right_on=["distrito", "Dia"], how="left")

        with h1:
            cols_m = ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_80m", "Media_Classe"]
            valid_cols = [c for c in cols_m if c in df_merged.columns]
            if len(valid_cols) > 1:
                fig5 = plt.figure(figsize=(5, 4))
                sns.heatmap(df_merged[valid_cols].dropna().corr(), annot=True, cmap="coolwarm", fmt=".2f", annot_kws={"size": 7})
                plt.title("Meteo vs Classe", fontsize=9)
                plt.xticks(fontsize=7, rotation=45)
                plt.yticks(fontsize=7)
                st.pyplot(fig5)

        with h2:
            col_meteo_jup = ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_80m", "uv_index_max"]
            m_p = [c for c in col_meteo_jup if c in df_merged.columns]
            p_p = [c for c in poluentesclean if c in df_merged.columns]
            if m_p and p_p:
                corr_sub = df_merged[m_p + p_p].corr().loc[m_p, p_p]
                fig6 = plt.figure(figsize=(6, 4))
                sns.heatmap(corr_sub, annot=True, fmt=".2f", cmap="coolwarm", annot_kws={"size": 7})
                plt.title("Meteo vs Poluentes (Detalhe)", fontsize=9)
                plt.xticks(fontsize=7, rotation=45)
                plt.yticks(fontsize=7)
                st.pyplot(fig6)
# ==============================================================================
# 3. MACHINE LEARNING (BASE)
# ==============================================================================
elif section == "Machine Learning (Base)":
    st.header("Machine Learning (Base - Meteo Simples)")
    
    if df_meteo is None:
        st.error("Sem dados de meteorologia.")
        st.stop()
        
    df_ar_ml = df_ar.rename(columns={'Data': 'date', 'Distrito': 'distrito'})
    df_ar_ml['date'] = df_ar_ml['date'].dt.normalize()
    
    distritos_validos = df_ar_ml['distrito'].unique()
    df_meteo_filtrado = df_meteo[df_meteo['distrito'].isin(distritos_validos)].copy()
    df_meteo_filtrado['date'] = df_meteo_filtrado['date'].dt.normalize()
    df_meteo_filtrado = df_meteo_filtrado.groupby(['date', 'distrito']).mean(numeric_only=True).reset_index()
    
    df_merged = pd.merge(df_ar_ml, df_meteo_filtrado, on=['date', 'distrito'], how='inner')
    df_Model = df_merged.dropna().copy()
    
    dfL = df_Model[df_Model["distrito"] == "Lisboa"].copy()
    dfL = dfL.sort_values("date").reset_index(drop=True)
    
    if dfL.empty:
        st.warning("Sem dados combinados.")
    else:
        st.subheader("Dataset (dfL)")
        st.dataframe(dfL)

        X_cols = [
            "rain", "temperature_2m", "relative_humidity_2m",
            "temperature_80m", "wind_speed_80m", "wind_direction_80m",
            "temperature_2m_max", "temperature_2m_min"
        ]
        Y_cols = ["O3", "NO2", "SO2", "PM10", "PM2.5"]
        
        if st.button("Treinar Modelos (Base)"):
            results = []
            param_grids = {
                "RandomForest": {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
                "LightGBM": {"n_estimators": [100, 200], "num_leaves": [31, 50], "learning_rate": [0.05, 0.1]},
                "MLP": {"hidden_layer_sizes": [(64,), (64,32)], "alpha": [0.0001, 0.001], "max_iter": [300, 500]}
            }
            models = {
                "RandomForest": RandomForestRegressor(random_state=42),
                "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
                "MLP": MLPRegressor(random_state=42)
            }
            prog = st.progress(0)
            
            for i, target in enumerate(Y_cols):
                if target not in dfL.columns: continue
                y = dfL[target].dropna()
                X = dfL[X_cols].loc[y.index]
                
                imputer = SimpleImputer(strategy="mean")
                X_imp = imputer.fit_transform(X)
                X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, shuffle=False)
                
                for name, model in models.items():
                    try:
                        grid = GridSearchCV(model, param_grids[name], cv=3, scoring="neg_mean_absolute_error", n_jobs=1)
                        grid.fit(X_train, y_train)
                        y_pred = grid.best_estimator_.predict(X_test)
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        results.append({"Poluente": target, "Modelo": name, "BestParams": str(grid.best_params_), "MAE": mae, "R2": r2})
                    except Exception as e:
                        st.write(f"Erro em {name}: {e}")
                prog.progress((i+1)/len(Y_cols))
            
            st.dataframe(pd.DataFrame(results))

# ==============================================================================
# 4. MACHINE LEARNING (AVANÇADO)
# ==============================================================================
elif section == "Machine Learning (Avançado)":
    st.header("Machine Learning (Avançado)")
    
    if df_meteo is None: st.stop()
        
    df_ar_ml = df_ar.rename(columns={'Data': 'date', 'Distrito': 'distrito'})
    df_ar_ml['date'] = df_ar_ml['date'].dt.normalize()
    
    distritos_validos = df_ar_ml['distrito'].unique()
    df_meteo_filtrado = df_meteo[df_meteo['distrito'].isin(distritos_validos)].copy()
    df_meteo_filtrado['date'] = df_meteo_filtrado['date'].dt.normalize()
    df_meteo_filtrado = df_meteo_filtrado.groupby(['date', 'distrito']).mean(numeric_only=True).reset_index()
    
    df_merged = pd.merge(df_ar_ml, df_meteo_filtrado, on=['date', 'distrito'], how='inner')
    df_Model = df_merged.dropna().copy()
    
    dfL = df_Model[df_Model["distrito"] == "Lisboa"].copy()
    dfL = dfL.sort_values("date").reset_index(drop=True)
    
    Y_cols = ["O3", "NO2", "SO2", "PM10", "PM2.5"]
    
    for col in Y_cols:
        if col in dfL.columns:
            dfL[f"{col}_lag1"] = dfL[col].shift(1)
            dfL[f"{col}_lag2"] = dfL[col].shift(2)
            dfL[f"{col}_roll3"] = dfL[col].rolling(3).mean()
            
    dfL["month"] = dfL["date"].dt.month
    dfL["weekday"] = dfL["date"].dt.weekday
    
    st.subheader("Dataset:")
    st.dataframe(dfL.head())
    
    X_cols_base = [
        "rain", "temperature_2m", "relative_humidity_2m",
        "temperature_80m", "wind_speed_80m", "wind_direction_80m",
        "temperature_2m_max", "temperature_2m_min"
    ]
    X_cols_lags = [f"{c}_lag1" for c in Y_cols] + [f"{c}_lag2" for c in Y_cols] + [f"{c}_roll3" for c in Y_cols]
    X_cols = X_cols_base + X_cols_lags + ["month", "weekday"]
    X_cols = [c for c in X_cols if c in dfL.columns]

    if st.button("Treinar Modelos (Avançado)"):
        results1 = []
        param_grids = {
            "RandomForest": {"n_estimators": [100, 200], "max_depth": [5, 10, None]},
            "LightGBM": {"n_estimators": [100, 200], "num_leaves": [31, 50], "learning_rate": [0.05, 0.1]},
            "MLP": {"hidden_layer_sizes": [(64,), (64,32)], "alpha": [0.0001, 0.001], "max_iter": [300, 500]}
        }
        models = {
            "RandomForest": RandomForestRegressor(random_state=42),
            "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
            "MLP": MLPRegressor(random_state=42)
        }
        tscv = TimeSeriesSplit(n_splits=3)
        prog = st.progress(0)
        
        for i, target in enumerate(Y_cols):
            if target not in dfL.columns: continue
            
            y = dfL[target].dropna()
            X = dfL[X_cols].loc[y.index]
            
            imputer = SimpleImputer(strategy="mean")
            X_imp = imputer.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, shuffle=False)
            
            for name, model in models.items():
                try:
                    grid = GridSearchCV(model, param_grids[name], cv=tscv, scoring="neg_mean_absolute_error", n_jobs=1)
                    grid.fit(X_train, y_train)
                    y_pred = grid.best_estimator_.predict(X_test)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    results1.append({"Poluente": target, "Modelo": name, "BestParams": str(grid.best_params_), "MAE": mae, "R2": r2})
                except Exception as e:
                    st.write(f"Erro em {name}: {e}")
            prog.progress((i+1)/len(Y_cols))
        
        st.dataframe(pd.DataFrame(results1))

# ==============================================================================
# 5. CLASSIFICAÇÃO
# ==============================================================================
elif section == "Classificação (Prever Classes)":
    st.header("Previsão de Classes (Classificação)")
    if df_meteo is None: st.stop()

    # 1. Preparação (Merge inicial)
    df_ar_ml = df_ar.rename(columns={'Data': 'date', 'Distrito': 'distrito'})
    df_ar_ml['date'] = df_ar_ml['date'].dt.normalize()
    distritos_validos = df_ar_ml['distrito'].unique()
    df_meteo_filtrado = df_meteo[df_meteo['distrito'].isin(distritos_validos)].copy()
    df_meteo_filtrado['date'] = df_meteo_filtrado['date'].dt.normalize()
    df_meteo_filtrado = df_meteo_filtrado.groupby(['date', 'distrito']).mean(numeric_only=True).reset_index()
    df_merged = pd.merge(df_ar_ml, df_meteo_filtrado, on=['date', 'distrito'], how='inner')
    df_Model = df_merged.dropna().copy()
    
    dfL = df_Model[df_Model["distrito"] == "Lisboa"].copy()
    dfL = dfL.sort_values("date").reset_index(drop=True)

    # 2. FEATURE ENGINEERING (ESSENCIAL PARA IGUALAR O AVANÇADO)
    Y_cols_reg = ["O3", "NO2", "SO2", "PM10", "PM2.5"]
    for col in Y_cols_reg:
        if col in dfL.columns:
            dfL[f"{col}_lag1"] = dfL[col].shift(1)
            dfL[f"{col}_lag2"] = dfL[col].shift(2)
            dfL[f"{col}_roll3"] = dfL[col].rolling(3).mean()
            
    dfL["month"] = dfL["date"].dt.month
    dfL["weekday"] = dfL["date"].dt.weekday

    # 3. DEFINIR X_COLS COMPLETO (INCLUINDO LAGS) - CORREÇÃO CRÍTICA AQUI
    X_cols_base = [
        "rain", "temperature_2m", "relative_humidity_2m",
        "temperature_80m", "wind_speed_80m", "wind_direction_80m",
        "temperature_2m_max", "temperature_2m_min"
    ]
    X_cols_lags = [f"{c}_lag1" for c in Y_cols_reg] + [f"{c}_lag2" for c in Y_cols_reg] + [f"{c}_roll3" for c in Y_cols_reg]
    X_cols = X_cols_base + X_cols_lags + ["month", "weekday"]
    X_cols = [c for c in X_cols if c in dfL.columns]

    df_class = dfL.copy()
    targets_class = ["PM10_classe", "PM2.5_classe", "NO2_classe", "O3_classe", "SO2_classe"]
    
    st.subheader("Dataset para Classificação (Com Features Avançadas)")
    cols_to_show = ["PM10_classe"] + X_cols[:5] + [c for c in X_cols if "lag1" in c]
    st.dataframe(df_class[cols_to_show].head())

    if st.button("Treinar Classificadores"):
        results_class = []
        tscv = TimeSeriesSplit(n_splits=5)
        
        # DEFINIR RANDOM STATE=42 EM TODOS OS MODELOS
        models = [
            (RandomForestClassifier(random_state=42), "RandomForest"),
            (LogisticRegression(max_iter=500, random_state=42), "LogisticRegression"),
            (SVC(kernel="rbf", random_state=42), "SVM"),
            (GaussianNB(), "NaiveBayes"),
            (KNeighborsClassifier(n_neighbors=5), "KNN"),
            (GradientBoostingClassifier(random_state=42), "GradientBoosting"),
            (DecisionTreeClassifier(random_state=42), "DecisionTree"),
            (ExtraTreesClassifier(random_state=42), "ExtraTrees"),
            (MLPClassifier(max_iter=500, random_state=42), "MLP")
        ]
        
        prog = st.progress(0)
        
        for i, target in enumerate(targets_class):
            if target not in df_class.columns: continue
            
            y = df_class[target].dropna()
            
            # PREENCHIMENTO DE NULOS COM FFILL/BFILL (IGUAL AO SNIPPET)
            X = df_class[X_cols].loc[y.index].fillna(method="ffill").fillna(method="bfill")
            
            # Verificar se ainda existem NaNs e preencher com média (segurança)
            if X.isna().any().any():
                 imputer = SimpleImputer(strategy="mean")
                 X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

            for clf, name in models:
                try:
                    accs, f1s = [], []
                    for train_idx, test_idx in tscv.split(X):
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                        
                        # LOGICA CRITICA: SE SÓ TEM 1 CLASSE, PULA O FOLD
                        if len(np.unique(y_train)) < 2: continue
                        
                        y_train = y_train.astype(int)
                        y_test = y_test.astype(int)
                        
                        clf.fit(X_train, y_train)
                        y_pred = clf.predict(X_test)
                        
                        accs.append(accuracy_score(y_test, y_pred))
                        f1s.append(f1_score(y_test, y_pred, average="weighted"))
                    
                    # LOGICA CRITICA: SE ACCS VAZIO (TODOS PULADOS), RETORNA NAN
                    if not accs:
                        final_acc = np.nan
                        final_f1 = np.nan
                    else:
                        final_acc = np.mean(accs)
                        final_f1 = np.mean(f1s)

                    results_class.append({
                        "Classe": target, "Modelo": name, 
                        "Accuracy": final_acc, "F1": final_f1
                    })
                except: pass
            prog.progress((i+1)/len(targets_class))
        
        st.dataframe(pd.DataFrame(results_class))

# ==============================================================================
# 6. SVR 
# ==============================================================================
elif section == "SVR":
    st.header("SVR")
    
    if df_meteo is None: st.stop()
    
    df_ar_ml = df_ar.rename(columns={'Data': 'date', 'Distrito': 'distrito'})
    df_ar_ml['date'] = df_ar_ml['date'].dt.normalize()
    
    distritos_validos = df_ar_ml['distrito'].unique()
    df_meteo_filtrado = df_meteo[df_meteo['distrito'].isin(distritos_validos)].copy()
    df_meteo_filtrado['date'] = df_meteo_filtrado['date'].dt.normalize()
    df_meteo_filtrado = df_meteo_filtrado.groupby(['date', 'distrito']).mean(numeric_only=True).reset_index()
    
    df_merged = pd.merge(df_ar_ml, df_meteo_filtrado, on=['date', 'distrito'], how='inner')
    df_Model = df_merged.dropna().copy()
    
    dfL = df_Model[df_Model["distrito"] == "Lisboa"].copy()
    dfL = dfL.sort_values("date").reset_index(drop=True)
    
    df_class = dfL.copy()
    
    if not df_class.empty:
        df_ar_svr = df_class.copy()
        for lag in range(1, 8):
            df_ar_svr[f"lag{lag}"] = df_ar_svr["Media_Classe"].shift(lag)
        
        df_ar_svr = df_ar_svr.dropna()
        
        st.subheader("Dataframe usado no SVR")
        st.dataframe(df_ar_svr)
        
        X = df_ar_svr[[f"lag{i}" for i in range(1, 8)]]
        y = df_ar_svr["Media_Classe"]
        
        model_ar = SVR(C=10, epsilon=0.1, gamma=0.01)
        model_ar.fit(X, y)
        y_pred = model_ar.predict(X)
        
        st.write("### Resultados SVR")
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mean_absolute_error(y, y_pred):.4f}")
        c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y, y_pred)):.4f}")
        c3.metric("R2", f"{r2_score(y, y_pred):.4f}")
        
        dates = df_class["date"].iloc[len(df_class)-len(y_pred):]
        real_values = df_class["Media_Classe"].iloc[len(df_class)-len(y_pred):]
        
        fig_svr = go.Figure()
        fig_svr.add_trace(go.Scatter(x=dates, y=real_values, mode="lines", name="Real", line=dict(color="blue")))
        fig_svr.add_trace(go.Scatter(x=dates, y=y_pred, mode="lines", name="Previsto (SVR)", line=dict(color="red")))
        fig_svr.update_layout(title="SVR Autoregressivo - Real vs Previsto", xaxis_title="Data", template="plotly_white")
        st.plotly_chart(fig_svr, use_container_width=True)

        # --- PREVISÃO 7 DIAS ---
        st.subheader("Previsão para os Próximos 7 Dias")
        
        last_window = list(df_class["Media_Classe"].iloc[-7:].values)
        future_preds = []
        last_date = df_class["date"].max()
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 8)]

        for _ in range(7):
            features = [last_window[-i] for i in range(1, 8)]
            pred = model_ar.predict([features])[0]
            future_preds.append(pred)
            last_window.append(pred)

        df_future = pd.DataFrame({"Data": future_dates, "Previsão (Media_Classe)": future_preds})
        st.dataframe(df_future)

        fig_future = go.Figure()
        recent_hist = df_class.iloc[-30:]
        fig_future.add_trace(go.Scatter(x=recent_hist["date"], y=recent_hist["Media_Classe"], mode="lines+markers", name="Histórico", line=dict(color="blue")))
        fig_future.add_trace(go.Scatter(x=df_future["Data"], y=df_future["Previsão (Media_Classe)"], mode="lines+markers", name="Previsão 7 Dias", line=dict(color="green", dash="dash")))
        fig_future.update_layout(title="Previsão de Qualidade do Ar (Próximos 7 Dias)", xaxis_title="Data", yaxis_title="Media Classe")
        st.plotly_chart(fig_future, use_container_width=True)

    else:
        st.warning("Sem dados suficientes para Lisboa.")

# ==============================================================================
# 7. CONCLUSÕES
# ==============================================================================
elif section == "Conclusão":
    st.header("Conclusões e Trabalho Futuro")
    
    st.write("Abaixo apresentamos uma síntese dos principais resultados alcançados, bem como as limitações e oportunidades de melhoria identificadas.")

    # 1. PRÓS E CONTRAS 
    st.subheader("Prós e Contras da Abordagem")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("**Pontos Fortes (Prós)**")
        st.markdown("""
        * **Recolha Automatizada (API):** A integração com APIs permite recolher e guardar dados de forma automática e muito mais acessível.
        * **Compreensão via EDA:** A Análise Exploratória de Dados permitiu visualizar o comportamento da qualidade do ar e identificar a sua relação com diversas variáveis.
        * **Engenharia de Features:** A criação de *Lags* e médias móveis melhorou significativamente a capacidade preditiva em comparação com o uso isolado da meteorologia.
        * **Diversidade de Modelos:** Testámos desde modelos lineares a *ensembles* (LightGBM, Random Forest) e redes neuronais (MLP), identificando o melhor ajuste para cada poluente.
        * **Previsão Recursiva:** A implementação do SVR autoregressivo permite gerar cenários futuros (7 dias) úteis para planeamento.
        * **Interface Interativa:** O dashboard permite explorar visualmente os dados e resultados de forma intuitiva.
        """)
        
    with col2:
        st.error("**Limitações (Contras)**")
        st.markdown("""
        * **Dimensão do Dataset:** O histórico temporal de 2025 é curto, o que limita a capacidade dos modelos de aprenderem padrões sazonais de longo prazo.
        * **Classes Desequilibradas:** Alguns poluentes (como SO2 e PM2.5) têm pouca variabilidade nas classes de qualidade, dificultando o treino de classificadores (muitos folds ignorados).
        * **Exclusão de Variáveis Relevantes:** Algumas variáveis que demonstraram forte relação com a qualidade do ar em 2023 tiveram de ser excluídas do modelo, uma vez que não existem dados equivalentes ou atualizados para 2025.
        * **Dependência de Dados Recentes:** A utilização de *Lags* limita a previsão ao curto prazo, pois o modelo necessita dos dados da semana anterior para projetar o futuro, inviabilizando previsões de longo prazo.
        """)

    st.divider()

    # 2. TRABALHO FUTURO 
    st.subheader("Trabalho Futuro")
    st.markdown("""

    * **1. Integração de Novas Features existentes para 2023:** Através do EDA onde relacionamos os dados de 2023 com 2025 foi possível ver relações com features que não foram utilizadas no nosso modelo de previsão como por exemplo os fogos. 
    * **2. Monitorizar features de forma a obter dados :** Embora não esteja no nosso EDA, devido à não existir datasets disponíveis existem features que seriam interessantes de explorar tal como o trânsito nas cidades.
    * **3. Continuar a monitorização as features que usamos:** Embora não tenhamos conseguido obter previsões com bons resultados para todos os componentes acreditamos que seja devido também à dimensão do nosso dataset, o estudo continuo e expansão dos dataset é essencial para conseguir melhorar os resultados. 
    * **4. Expansão Geográfica:** Alargar a previsão detalhada a outros distritos além de Lisboa, permitindo uma comparação regional mais robusta.
    * **5. Previsão para componentes:** Visto que neste trabalho o nosso foco foi estudar a variavél alvo (Média Qualidade do Ar) e não obtivos bons resultados nas métricas para todos os componentes acabamos por não fazer a previsão para os  mesmos.
    """)






