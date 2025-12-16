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

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="Replica Notebook", layout="wide")
st.title("Notebook: Análise e Modelação da Qualidade do Ar")

# --- INICIALIZAÇÃO SEGURA DE VARIÁVEIS (Para evitar NameError) ---
df_ar = pd.DataFrame()
df2023_clean = pd.DataFrame()
df_meteo = pd.DataFrame()
df_combinadometeoqualar = pd.DataFrame()

# ==============================================================================
# 1. FUNÇÕES DE PROCESSAMENTO
# ==============================================================================

@st.cache_data
def carregar_dados():
    # Variáveis locais
    data_2025 = None
    data_2023 = None
    data_meteo = None
    distritos_target = ['Aveiro', 'Lisboa', 'Açores', 'Setúbal', 'Leiria', 'Madeira', 'Santarém']

    # --- A. DADOS 2025 ---
    try:
        df = pd.read_excel("QualidadeAr2.xlsx")
        # Replica limpeza notebook
        if 'Coluna1' in df.columns: df = df.rename(columns={"Coluna1": "Distrito"})
        if "Tipo" in df.columns: df = df.drop(["Tipo", "Zona"], axis=1)
        
        df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')
        df['Semana'] = df['Data'].dt.isocalendar().week
        df['Ano'] = df['Data'].dt.year
        
        cols_num = ['C6H6', 'CO', 'NO2', 'O3', 'PM2.5', 'PM10', 'SO2']
        for c in cols_num:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

        # Imputação (Média Semanal)
        df_medias = df.groupby(['Distrito', 'Ano', 'Semana'])[cols_num].mean().reset_index()
        df = df.merge(df_medias, on=['Distrito', 'Ano', 'Semana'], suffixes=('', '_media'), how='left')
        for c in cols_num:
            if f'{c}_media' in df.columns: df[c] = df[c].fillna(df[f'{c}_media'])
        
        # Agrupar
        df_agrupado = df.groupby(["Data", "Distrito", "Semana", "Ano"])[cols_num].mean().reset_index()

        # Classificação
        intervalos = {
            "PM10": [(101, 1200), (51, 100), (36, 50), (21, 35), (0, 20)],
            "PM2.5": [(51, 800), (26, 50), (21, 25), (11, 20), (0, 10)],
            "NO2": [(401, 1000), (201, 400), (101, 200), (41, 100), (0, 40)],
            "O3": [(241, 600), (181, 240), (101, 180), (81, 100), (0, 80)],
            "SO2": [(501, 1250), (351, 500), (201, 350), (101, 200), (0, 100)]
        }

        def classificar(v, lims):
            if pd.isna(v): return np.nan
            dists = []
            for i, (mn, mx) in enumerate(lims):
                centro = (mn + mx) / 2
                dists.append((abs(v - centro), i + 1))
            return min(dists)[1]

        for p in intervalos:
            df_agrupado[f'{p}_classe'] = df_agrupado[p].apply(lambda x: classificar(x, intervalos[p]))
            
        data_2025 = df_agrupado[df_agrupado['Distrito'].isin(distritos_target)].copy()
        
        cls_cols = [c for c in data_2025.columns if c.endswith('_classe')]
        data_2025['Media_Classe'] = data_2025[cls_cols].mean(axis=1)
        
    except Exception as e:
        st.error(f"Erro processamento 2025: {e}")

    # --- B. DADOS 2023 ---
    try:
        df = pd.read_excel("Qualar2023.xlsx")
        df.columns = df.columns.str.strip()
        if 'Local' in df.columns: df = df.drop('Local', axis=1)
        if 'Data-Hora' in df.columns:
            df['Data-Hora'] = pd.to_datetime(df['Data-Hora'])
            df['Semana'] = df['Data-Hora'].dt.isocalendar().week
            df['Ano'] = df['Data-Hora'].dt.year
            
        cols_num = ['C6H6', 'CO', 'NO2', 'O3', 'PM2.5', 'PM10', 'SO2']
        for c in cols_num:
            if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

        df_sem = df.groupby(['Distrito', 'Ano', 'Semana'])[cols_num].mean(numeric_only=True).reset_index()
        df = df.merge(df_sem, on=['Distrito', 'Ano', 'Semana'], suffixes=('', '_media'), how='left')
        for c in cols_num:
            if f'{c}_media' in df.columns: df[c] = df[c].fillna(df[f'{c}_media'])
            
        df_agrupado = df.groupby(["Distrito", "Data-Hora", "Semana", "Ano"])[cols_num].mean(numeric_only=True).reset_index()
        
        for p in intervalos:
            df_agrupado[f'{p}_classe'] = df_agrupado[p].apply(lambda x: classificar(x, intervalos[p]))
            
        cls_cols = [c for c in df_agrupado.columns if c.endswith('_classe')]
        df_agrupado['Media_Classe'] = df_agrupado[cls_cols].mean(axis=1)
        
        data_2023 = df_agrupado[df_agrupado['Distrito'].isin(distritos_target)].copy()
        data_2023['Data'] = pd.to_datetime(data_2023['Data-Hora']).dt.date
        data_2023['Data'] = pd.to_datetime(data_2023['Data'])
        
    except Exception as e:
        st.warning(f"Erro processamento 2023: {e}")

    # --- C. METEOROLOGIA (AQUI ESTAVA O ERRO) ---
    try:
        df = pd.read_csv("dataset_meteorologico_portugal.csv")
        
        # Truque para forçar leitura correta de datas mistas ou com TZ
        df['date'] = pd.to_datetime(df['date'], utc=True).dt.tz_localize(None)
        
        if 'distrito' in df.columns:
            df['distrito'] = df['distrito'].astype(str).str.strip().str.title()
            
        data_meteo = df
    except Exception as e:
        st.error(f"Erro crítico meteorologia: {e}")

    return data_2025, data_2023, data_meteo

# Executar Carregamento
df_ar, df2023_clean, df_meteo = carregar_dados()

if df_ar is None:
    st.stop()

poluentesclean = ['NO2', 'O3', 'PM2.5', 'PM10', 'SO2']

# ==============================================================================
# 2. VISUALIZAÇÕES
# ==============================================================================
st.write("--- 2. Análise Exploratória ---")

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
    # Filtro exato de datas para bater certo
    datas_2025 = df_ar["Data"].dt.strftime("%m-%d").unique()
    df2023_periodo = df2023_clean[df2023_clean["Data"].dt.strftime("%m-%d").isin(datas_2025)].copy()
    
    df2023_periodo["Ano"] = 2023
    df2025_periodo = df_ar.copy()
    df2025_periodo["Ano"] = 2025
    
    # Uniformizar distritos para o merge
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
    
    st.subheader("Variação Percentual")
    df23_avg = df2023_periodo.groupby("Distrito")[poluentesclean].mean().reset_index()
    df25_avg = df2025_periodo.groupby("Distrito")[poluentesclean].mean().reset_index()
    comparacao = pd.merge(df23_avg, df25_avg, on="Distrito", suffixes=("_2023", "_2025"))
    
    fig3, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes = axes.flatten()
    for i, p in enumerate(poluentesclean):
        comparacao[f"{p}_var"] = ((comparacao[f"{p}_2025"] - comparacao[f"{p}_2023"]) / comparacao[f"{p}_2023"]) * 100
        comp_sorted = comparacao.sort_values(f"{p}_var", ascending=False)
        axes[i].bar(comp_sorted["Distrito"], comp_sorted[f"{p}_var"])
        axes[i].axhline(0, color="gray", linestyle="--")
        axes[i].set_title(f"{p} (2023 -> 2025)")
        axes[i].tick_params(axis='x', rotation=45)
    for j in range(len(poluentesclean), len(axes)):
        fig3.delaxes(axes[j])
    plt.tight_layout()
    st.pyplot(fig3)
    
    st.subheader("Média Geral da Classe")
    med_23 = df2023_periodo['Media_Classe'].mean()
    med_25 = df2025_periodo['Media_Classe'].mean()
    media_df = pd.DataFrame({'Ano': ['2023', '2025'], 'Media': [med_23, med_25]})
    
    fig4 = plt.figure(figsize=(6,5))
    ax = sns.barplot(data=media_df, x='Ano', y='Media', palette=['#FFA500', '#1F77B4'])
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width()/2., p.get_height()), ha='center', va='bottom')
    plt.ylim(0, 5.5)
    st.pyplot(fig4)

# ==============================================================================
# 3. MACHINE LEARNING
# ==============================================================================
st.write("--- 3. Machine Learning ---")

# Merge com Meteorologia (Com verificação de segurança)
if df_meteo is not None and not df_meteo.empty:
    
    # Preparar chaves de merge
    df_ar_m = df_ar.copy()
    df_ar_m["Dia"] = df_ar_m["Data"].dt.date
    df_meteo["Dia"] = df_meteo["date"].dt.date
    
    # MERGE VITAL
    df_combinadometeoqualar = pd.merge(
        df_ar_m, df_meteo, left_on=["Distrito", "Dia"], right_on=["distrito", "Dia"], how="left"
    )
    
    # Heatmap Correlação
    cols_meteo = ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_80m", "Media_Classe"]
    cols_validas = [c for c in cols_meteo if c in df_combinadometeoqualar.columns]
    
    if len(cols_validas) > 1:
        st.subheader("Correlações")
        df_corr = df_combinadometeoqualar[cols_validas].dropna()
        if not df_corr.empty:
            fig5 = plt.figure(figsize=(10, 8))
            sns.heatmap(df_corr.corr(), annot=True, cmap="coolwarm", fmt=".2f")
            st.pyplot(fig5)

    # --------------------------------------------------------------------------
    # MODELOS PREDICTIVOS
    # --------------------------------------------------------------------------
    df_Model = df_combinadometeoqualar.dropna().copy()
    
    # Filtro Lisboa e Ordenação
    dfL = df_Model[df_Model["distrito"] == "Lisboa"].copy()
    dfL = dfL.sort_values("date").reset_index(drop=True)
    
    if not dfL.empty:
        
        # A. REGRESSÃO (Réplica do Snippet do Notebook - RESULTADOS MAUS ESPERADOS)
        st.info("Clique para treinar modelos (pode demorar)")
        if st.button("Treinar Modelos"):
            
            # --- 1. Tabela Geral (Meteo Simples -> Poluente) ---
            # Features EXATAS do teu snippet (sem lags)
            X_cols = [
                "rain", "temperature_2m", "relative_humidity_2m",
                "temperature_80m", "wind_speed_80m", "wind_direction_80m",
                "temperature_2m_max", "temperature_2m_min"
            ]
            Y_cols = ["O3", "NO2", "SO2", "PM10", "PM2.5"]
            
            # Modelos
            param_grids = {
                "RandomForest": {"n_estimators": [100], "max_depth": [5, 10, None]},
                "LightGBM": {"n_estimators": [100], "learning_rate": [0.05, 0.1]},
                "MLP": {"hidden_layer_sizes": [(64,)], "alpha": [0.001], "max_iter": [200]}
            }
            models = {
                "RandomForest": RandomForestRegressor(random_state=42),
                "LightGBM": LGBMRegressor(random_state=42, verbose=-1),
                "MLP": MLPRegressor(random_state=42)
            }
            
            results = []
            progress = st.progress(0)
            
            for i, target in enumerate(Y_cols):
                if target not in dfL.columns: continue
                
                y = dfL[target]
                X = dfL[X_cols] # Meteo base apenas
                
                # Split sem shuffle
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
                
                for name, model in models.items():
                    try:
                        grid = GridSearchCV(model, param_grids[name], cv=3, scoring="neg_mean_absolute_error", n_jobs=1)
                        grid.fit(X_train, y_train)
                        y_pred = grid.best_estimator_.predict(X_test)
                        
                        mae = mean_absolute_error(y_test, y_pred)
                        r2 = r2_score(y_test, y_pred)
                        
                        results.append({
                            "Poluente": target, "Modelo": name, 
                            "BestParams": str(grid.best_params_), "MAE": mae, "R2": r2
                        })
                    except: pass
                progress.progress((i+1)/len(Y_cols))
            
            st.write("### Resultados Regressão (Meteorologia Apenas)")
            st.dataframe(pd.DataFrame(results))
            
            # --- 2. SVR Autoregressivo (Previsão com Histórico) ---
            st.write("---")
            st.subheader("SVR Autoregressivo (Media_Classe)")
            
            # Aqui criamos lags porque o modelo SVR precisa deles
            df_class = dfL.copy()
            for lag in range(1, 8):
                df_class[f"lag{lag}"] = df_class["Media_Classe"].shift(lag)
            
            df_class = df_class.dropna()
            
            if not df_class.empty:
                X_svr = df_class[[f"lag{i}" for i in range(1, 8)]]
                y_svr = df_class["Media_Classe"]
                
                model_ar = SVR(C=10, epsilon=0.1, gamma=0.01)
                model_ar.fit(X_svr, y_svr)
                y_pred_svr = model_ar.predict(X_svr)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("MAE", f"{mean_absolute_error(y_svr, y_pred_svr):.4f}")
                c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y_svr, y_pred_svr)):.4f}")
                c3.metric("R2", f"{r2_score(y_svr, y_pred_svr):.4f}")
                
                # Gráfico Plotly
                fig_svr = go.Figure()
                fig_svr.add_trace(go.Scatter(x=df_class["date"], y=y_svr, mode="lines", name="Real", line=dict(color="navy")))
                fig_svr.add_trace(go.Scatter(x=df_class["date"], y=y_pred_svr, mode="lines", name="Previsto (SVR)", line=dict(color="red")))
                fig_svr.update_layout(title="SVR Autoregressivo", xaxis_title="Data", template="plotly_white")
                st.plotly_chart(fig_svr, use_container_width=True)

    else:
        st.warning("Sem dados para Lisboa (verifique nomes dos distritos).")
else:
    st.error("Erro: Não foi possível cruzar os dados. Verifique se o ficheiro de meteorologia carregou corretamente.")
