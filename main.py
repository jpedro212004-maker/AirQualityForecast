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
# CONFIG
# =============================
st.set_page_config(page_title="Qualidade do Ar em Portugal", layout="wide")
st.title("🌍 Qualidade do Ar e Meteorologia em Portugal")

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
        # CORREÇÃO CRÍTICA PARA ERRO .DT
        df_meteo["date"] = pd.to_datetime(df_meteo["date"], utc=True)
        df_meteo["date"] = df_meteo["date"].dt.tz_localize(None)
        
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
        # Filtro de datas para igualar Notebook
        datas_2025 = df_ar["Data"].dt.strftime("%m-%d").unique()
        df23_p = df2023_clean[df2023_clean["Data"].dt.strftime("%m-%d").isin(datas_2025)].copy()
        
        df23_p["Ano"] = 2023
        df25_p = df_ar.copy()
        df25_p["Ano"] = 2025
        # Uniformizar distritos
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
# 3. MACHINE LEARNING (IGUAL AO SNIPPET)
# ==============================================================================
elif section == "Machine Learning":
    st.header("🤖 Machine Learning (Lisboa)")
    st.markdown("Resultados da Regressão com GridSearch (sem Lags, apenas Meteo base).")
    
    if df_meteo is None:
        st.error("Sem dados de meteorologia.")
        st.stop()
        
    # --- PREPARAÇÃO DOS DADOS IGUAL AO NOTEBOOK ---
    # 1. Renomear colunas para uniformizar (Passo crucial do teu notebook)
    df_ar_ml = df_ar.rename(columns={'Data': 'date', 'Distrito': 'distrito'})
    
    # 2. Filtrar distritos válidos na meteo
    distritos_validos = df_ar_ml['distrito'].unique()
    df_meteo_filtrado = df_meteo[df_meteo['distrito'].isin(distritos_validos)]
    
    # 3. Tratamento de data e group by na meteo (conforme teu snippet)
    # Converter para diário (média por dia)
    df_meteo_filtrado['date'] = df_meteo_filtrado['date'].dt.floor('D')
    df_meteo_filtrado = df_meteo_filtrado.groupby(['date', 'distrito']).mean(numeric_only=True).reset_index()
    
    # 4. Merge pelos campos comuns: 'date' e 'distrito'
    df_merged = pd.merge(df_ar_ml, df_meteo_filtrado, on=['date', 'distrito'], how='inner')
    
    # 5. Drop NA
    df_Model = df_merged.dropna().copy()
    
    # 6. Filtrar Lisboa
    dfL = df_Model[df_Model["distrito"] == "Lisboa"].copy()
    dfL = dfL.sort_values("date").reset_index(drop=True)
    
    if dfL.empty:
        st.warning("Sem dados combinados para Lisboa.")
    else:
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
                "RandomForest": {
                    "n_estimators": [100, 200],
                    "max_depth": [5, 10, None]
                },
                "LightGBM": {
                    "n_estimators": [100, 200],
                    "num_leaves": [31, 50],
                    "learning_rate": [0.05, 0.1]
                },
                "MLP": {
                    "hidden_layer_sizes": [(64,), (64,32)],
                    "alpha": [0.0001, 0.001],
                    "max_iter": [300, 500]
                }
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
                X = dfL[X_cols].loc[y.index] # X SEM LAGS AQUI
                
                # Imputer (Como no notebook)
                imputer = SimpleImputer(strategy="mean")
                X_imp = imputer.fit_transform(X)
                
                # Split sem shuffle
                X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, shuffle=False)
                
                for name, model in models.items():
                    try:
                        # GridSearch igual ao notebook (cv=3)
                        grid = GridSearchCV(model, param_grids[name], cv=3, scoring="neg_mean_absolute_error", n_jobs=1)
                        grid.fit(X_train, y_train)
                        
                        best_model = grid.best_estimator_
                        y_pred = best_model.predict(X_test)
                        
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        results.append({
                            "Poluente": target, "Modelo": name, 
                            "BestParams": str(grid.best_params_), "MAE": mae, "R2": r2
                        })
                    except Exception as e:
                        st.write(f"Erro em {name}: {e}")
                prog.progress((i+1)/len(Y_cols))
            
            st.dataframe(pd.DataFrame(results))

# ==============================================================================
# 4. SVR AUTOREGRESSIVO (CORRIGIDO: SEM DEPENDÊNCIA DA METEOROLOGIA)
# ==============================================================================
elif section == "SVR Autoregressivo":
    st.header("📈 SVR Autoregressivo (Com Lags)")
    
    # EM VEZ DE USAR O MERGE, USAMOS DIRETAMENTE OS DADOS DO AR (df_ar)
    # Isto evita que falhas no ficheiro de meteorologia cortem os últimos dias de dados
    
    # 1. Filtro Lisboa e Ordenação
    # Nota: No df_ar original os nomes são 'Distrito' e 'Data' (com maiúscula)
    dfL = df_ar[df_ar["Distrito"] == "Lisboa"].copy()
    dfL = dfL.sort_values("Data")
    
    if not dfL.empty:
        # 2. Criar Lags (Igual ao notebook)
        df_class = dfL.copy()
        for lag in range(1, 8):
            df_class[f"lag{lag}"] = df_class["Media_Classe"].shift(lag)
        
        # 3. Drop NAs gerados pelos lags (perde-se a primeira semana, mas mantém-se o fim)
        df_class = df_class.dropna()
        
        # 4. Definir X e y
        X_svr = df_class[[f"lag{i}" for i in range(1, 8)]]
        y_svr = df_class["Media_Classe"]
        
        # 5. Treino
        # Parâmetros exatos do teu notebook
        model_ar = SVR(C=10, epsilon=0.1, gamma=0.01)
        model_ar.fit(X_svr, y_svr)
        y_pred = model_ar.predict(X_svr)
        
        # 6. Métricas
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mean_absolute_error(y_svr, y_pred):.4f}")
        c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_svr, y_pred)):.4f}")
        c3.metric("R2", f"{r2_score(y_svr, y_pred):.4f}")
        
        # 7. Gráfico
        fig_svr = go.Figure()
        fig_svr.add_trace(go.Scatter(x=df_class["Data"], y=y_svr, mode="lines", name="Real", line=dict(color="blue")))
        fig_svr.add_trace(go.Scatter(x=df_class["Data"], y=y_pred, mode="lines", name="Previsto (SVR)", line=dict(color="red")))
        
        fig_svr.update_layout(
            title="SVR - Real vs Previsto", 
            xaxis_title="Data", 
            yaxis_title="Media da Classe",
            template="plotly_white",
            hovermode="x unified"
        )
        st.plotly_chart(fig_svr, use_container_width=True)
    else:
        st.warning("Sem dados suficientes para Lisboa no ficheiro de Qualidade do Ar.")
