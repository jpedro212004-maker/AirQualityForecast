import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="Análise Qualidade Ar", layout="wide")
st.title("Representação da Análise de Dados (Notebook Replicado)")

# --- 1. FUNÇÕES AUXILIARES DE PROCESSAMENTO ---

def processar_dataset(df, ano):
    """
    Replica a lógica de limpeza e imputação (fillna) do notebook.
    """
    # 1. Limpeza básica
    if 'Coluna1' in df.columns: df = df.rename(columns={"Coluna1": "Distrito"})
    if 'Local' in df.columns: df = df.drop('Local', axis=1)
    df.columns = df.columns.str.strip()
    
    # Renomear data se necessário
    if 'Data-Hora' in df.columns: df = df.rename(columns={'Data-Hora': 'Data'})
    
    # Converter Data
    df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')
    df['Ano'] = df['Data'].dt.year
    df['Semana'] = df['Data'].dt.isocalendar().week
    
    # 2. Converter Poluentes para Numérico
    poluentes = ['C6H6', 'CO', 'NO2', 'O3', 'PM2.5', 'PM10', 'SO2']
    cols_existentes = [c for c in poluentes if c in df.columns]
    
    for p in cols_existentes:
        df[p] = pd.to_numeric(df[p], errors='coerce')
        
    # 3. IMPUTAÇÃO (CRUCIAL PARA OS VALORES BATEREM CERTO)
    if not cols_existentes:
        return df
        
    df_medias = df.groupby(['Distrito', 'Ano', 'Semana'])[cols_existentes].mean().reset_index()
    
    # Merge e Fillna
    df = df.merge(df_medias, on=['Distrito', 'Ano', 'Semana'], suffixes=('', '_media'), how='left')
    
    for p in cols_existentes:
        if f'{p}_media' in df.columns:
            df[p] = df[p].fillna(df[f'{p}_media'])
            
    return df

def classificar_proximo(valor, intervalos):
    if pd.isna(valor): return np.nan
    distancias = []
    for i, (minimo, maximo) in enumerate(intervalos):
        centro = (minimo + maximo) / 2
        distancias.append((abs(valor - centro), i + 1)) 
    return min(distancias)[1]

# --- 2. CARREGAMENTO ---
@st.cache_data
def load_data():
    data = {}
    
    # Carregar 2025
    try:
        df25 = pd.read_excel("QualidadeAr2.xlsx")
        df25 = processar_dataset(df25, 2025)
        data["ar_2025"] = df25
    except Exception as e:
        st.error(f"Erro 2025: {e}")

    # Carregar 2023
    try:
        df23 = pd.read_excel("Qualar2023.xlsx")
        df23 = processar_dataset(df23, 2023)
        data["ar_2023"] = df23
    except Exception as e:
        st.error(f"Erro 2023: {e}")
        
    # Carregar Meteo
    try:
        df_meteo = pd.read_csv("dataset_meteorologico_portugal.csv")
        # Forçar conversão de data aqui também para garantir
        df_meteo['date'] = pd.to_datetime(df_meteo['date'], errors='coerce')
        df_meteo['distrito'] = df_meteo['distrito'].str.strip().str.title()
        data["meteo"] = df_meteo
    except:
        pass

    return data

dados = load_data()
if not dados: st.stop()

df25_full = dados.get('ar_2025')
df23_full = dados.get('ar_2023')
df_meteo = dados.get('meteo')

# --- 3. LÓGICA DE FILTRO E CLASSES ---

distritos_desejados = ['Aveiro', 'Lisboa', 'Açores', 'Setúbal', 'Leiria', 'Madeira', 'Santarém']
poluentesclean = ['NO2', 'O3', 'PM2.5', 'PM10', 'SO2']

intervalos = {
    "PM10": [(101, 1200), (51, 100), (36, 50), (21, 35), (0, 20)],
    "PM2.5": [(51, 800), (26, 50), (21, 25), (11, 20), (0, 10)],
    "NO2": [(401, 1000), (201, 400), (101, 200), (41, 100), (0, 40)],
    "O3": [(241, 600), (181, 240), (101, 180), (81, 100), (0, 80)],
    "SO2": [(501, 1250), (351, 500), (201, 350), (101, 200), (0, 100)]
}

# 2025
df_ar = df25_full[df25_full['Distrito'].isin(distritos_desejados)].copy()
for p in intervalos:
    df_ar[f'{p}_classe'] = df_ar[p].apply(lambda x: classificar_proximo(x, intervalos[p]))

colunas_classes = [c for c in df_ar.columns if c.endswith('_classe')]
df_ar['Media_Classe'] = df_ar[colunas_classes].mean(axis=1)

# 2023
df23_clean = pd.DataFrame()
if df23_full is not None:
    df23_temp = df23_full[df23_full['Distrito'].isin(distritos_desejados)].copy()
    
    # Filtro de Data
    if not df_ar.empty:
        datas_2025 = df_ar["Data"].dt.strftime("%m-%d").unique()
        df23_clean = df23_temp[df23_temp["Data"].dt.strftime("%m-%d").isin(datas_2025)].copy()
        
        for p in intervalos:
            df23_clean[f'{p}_classe'] = df23_clean[p].apply(lambda x: classificar_proximo(x, intervalos[p]))
            
        colunas_classes_23 = [c for c in df23_clean.columns if c.endswith('_classe')]
        df23_clean['Media_Classe'] = df23_clean[colunas_classes_23].mean(axis=1)

# ==============================================================================
# VISUALIZAÇÕES
# ==============================================================================

st.header("1. Boxplots Componentes Diários 2025")
df_por_dia = df_ar.groupby("Data")[poluentesclean].mean().reset_index()

fig1, axes = plt.subplots(nrows=1, ncols=len(poluentesclean), figsize=(18, 5))
for i, col in enumerate(poluentesclean):
    sns.boxplot(y=df_por_dia[col], ax=axes[i], color="skyblue")
    axes[i].set_title(f'Boxplot de {col} em 2025')
    axes[i].set_ylabel("")
plt.tight_layout()
st.pyplot(fig1)

# ------------------------------------------------------------------------------

if not df23_clean.empty:
    st.header("2. Comparação Distribuição 2023 vs 2025")
    
    df23_viz = df23_clean[["Data", "Distrito", "Media_Classe"] + poluentesclean].copy()
    df23_viz["Ano"] = 2023
    
    df25_viz = df_ar[["Data", "Distrito", "Media_Classe"] + poluentesclean].copy()
    df25_viz["Ano"] = 2025
    
    df_comparacao = pd.concat([df23_viz, df25_viz], ignore_index=True)
    
    fig2, axes = plt.subplots(2, 3, figsize=(15,8))
    axes = axes.flatten()
    for i, p in enumerate(poluentesclean):
        sns.boxplot(data=df_comparacao, x='Ano', y=p, ax=axes[i], palette="Set2")
        axes[i].set_title(p)
    
    for j in range(len(poluentesclean), len(axes)):
        fig2.delaxes(axes[j])
            
    plt.suptitle("Distribuição dos poluentes — comparação 2023 vs 2025")
    plt.tight_layout()
    st.pyplot(fig2)

    # ------------------------------------------------------------------------------
    st.header("3. Variação Percentual por Distrito")
    
    df23_avg = df23_viz.groupby("Distrito")[poluentesclean].mean().reset_index()
    df25_avg = df25_viz.groupby("Distrito")[poluentesclean].mean().reset_index()
    comparacao = pd.merge(df23_avg, df25_avg, on="Distrito", suffixes=("_2023", "_2025"))
    
    for p in poluentesclean:
        comparacao[f"{p}_var_percent"] = ((comparacao[f"{p}_2025"] - comparacao[f"{p}_2023"]) / comparacao[f"{p}_2023"]) * 100
        
    fig3, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    axes = axes.flatten()
    for i, poluente in enumerate(poluentesclean):
        comparacao_sorted = comparacao.sort_values(f"{poluente}_var_percent", ascending=False)
        axes[i].bar(comparacao_sorted["Distrito"], comparacao_sorted[f"{poluente}_var_percent"])
        axes[i].axhline(0, color="gray", linestyle="--")
        axes[i].set_title(f"{poluente} (2023 -> 2025)")
        axes[i].set_ylabel("Variação (%)")
        axes[i].tick_params(axis='x', rotation=45)
        
    for j in range(len(poluentesclean), len(axes)):
        fig3.delaxes(axes[j])
            
    plt.tight_layout()
    st.pyplot(fig3)

    # ------------------------------------------------------------------------------
    st.header("4. Comparação Média Geral da Classe")
    
    media_geral_23 = df23_viz['Media_Classe'].mean()
    media_geral_25 = df25_viz['Media_Classe'].mean()
    
    media_geral_df = pd.DataFrame({
        'Ano': ['2023', '2025'],
        'Media_Classe': [media_geral_23, media_geral_25]
    })
    
    fig4 = plt.figure(figsize=(6,5))
    ax = sns.barplot(data=media_geral_df, x='Ano', y='Media_Classe', palette=['#FFA500', '#1F77B4'])
    
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()), 
                   ha='center', va='bottom', fontsize=12)
                   
    plt.ylim(0, 5.5)
    st.pyplot(fig4)

# ------------------------------------------------------------------------------

if df_meteo is not None:
    st.header("5. Correlações (Meteo vs Qualidade Ar)")
    
    # --- CORREÇÃO DO ERRO ---
    # Garantir que a coluna 'date' é datetime antes de usar .dt
    df_meteo['date'] = pd.to_datetime(df_meteo['date'], errors='coerce')
    
    # Preparar merge
    df_ar_corr = df_ar.copy()
    df_ar_corr["Dia"] = df_ar_corr["Data"].dt.date
    df_meteo["Dia"] = df_meteo["date"].dt.date 
    
    df_combinado = pd.merge(df_ar_corr, df_meteo, left_on=["Distrito", "Dia"], right_on=["distrito", "Dia"], how="left")
    
    # Heatmap 1
    cols_meteo = ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_80m", "Media_Classe"]
    # Dropna para garantir correlação válida
    df_corr1 = df_combinado[cols_meteo].dropna().corr()
    
    fig5 = plt.figure(figsize=(8, 6))
    sns.heatmap(df_corr1, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    st.pyplot(fig5)
    
    # Heatmap 2
    st.subheader("Correlação Meteo vs Componentes")
    cols_full = ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_80m"] + poluentesclean
    df_corr2 = df_combinado[cols_full].corr()
    
    meteo_vars = ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_80m"]
    corr_sub = df_corr2.loc[[c for c in meteo_vars if c in df_corr2.index], 
                            [c for c in poluentesclean if c in df_corr2.columns]]
    
    fig6 = plt.figure(figsize=(10, 6))
    sns.heatmap(corr_sub, annot=True, cmap="coolwarm", fmt=".2f", center=0)
    st.pyplot(fig6)

# ==============================================================================
# MACHINE LEARNING - SVR AUTOREGRESSIVO
# ==============================================================================

st.header("6. Machine Learning: SVR Autoregressivo")
st.markdown("Previsão para **Lisboa**")

df_lisboa = df_ar[df_ar['Distrito'] == 'Lisboa'].sort_values('Data').copy()

if len(df_lisboa) > 10:
    for lag in range(1, 8):
        df_lisboa[f"lag{lag}"] = df_lisboa["Media_Classe"].shift(lag)
    
    df_lisboa = df_lisboa.dropna()
    
    if not df_lisboa.empty:
        X = df_lisboa[[f"lag{i}" for i in range(1, 8)]]
        y = df_lisboa["Media_Classe"]
        
        try:
            model_ar = SVR(C=10, epsilon=0.1, gamma=0.01)
            model_ar.fit(X, y)
            y_pred = model_ar.predict(X)
            
            c1, c2, c3 = st.columns(3)
            c1.metric("MAE", f"{mean_absolute_error(y, y_pred):.4f}")
            c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y, y_pred)):.4f}")
            c3.metric("R2 Score", f"{r2_score(y, y_pred):.4f}")
            
            # Gráfico Plotly
            fig_svr = go.Figure()
            fig_svr.add_trace(go.Scatter(x=df_lisboa["Data"], y=y, mode="lines", name="Real", line=dict(color="navy")))
            fig_svr.add_trace(go.Scatter(x=df_lisboa["Data"], y=y_pred, mode="lines", name="Previsto (SVR)", line=dict(color="red")))
            
            fig_svr.update_layout(title="SVR Autoregressivo - Real vs Previsto", xaxis_title="Data", yaxis_title="Media_Classe", template="plotly_white")
            st.plotly_chart(fig_svr, use_container_width=True)
            
        except Exception as e:
            st.error(f"Erro ao treinar SVR: {e}")
else:
    st.warning("Dados insuficientes para Lisboa.")
