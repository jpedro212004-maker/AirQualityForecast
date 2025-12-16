import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

# --- CONFIGURAÇÃO ---
st.set_page_config(page_title="Análise Qualidade Ar", layout="wide")
st.title("Representação da Análise de Dados (Notebook)")

# --- 1. CARREGAMENTO E LIMPEZA (Conforme o teu código) ---
@st.cache_data
def load_and_process_data():
    data = {}
    
    # Colunas numéricas obrigatórias
    cols_num = ['NO2', 'O3', 'PM10', 'PM2.5', 'SO2', 'CO', 'C6H6']
    
    # --- CARREGAR 2025 ---
    try:
        df25 = pd.read_excel("QualidadeAr2.xlsx")
        if 'Coluna1' in df25.columns: df25 = df25.rename(columns={"Coluna1": "Distrito"})
        df25['Data'] = pd.to_datetime(df25['Data'], dayfirst=True, errors='coerce')
        df25['Ano'] = 2025
        # Conversão forçada para numérico (Lógica do teu código)
        for col in cols_num:
            if col in df25.columns: df25[col] = pd.to_numeric(df25[col], errors='coerce')
        data["ar_2025"] = df25
    except Exception as e:
        st.error(f"Erro 2025: {e}")
        return None

    # --- CARREGAR 2023 ---
    try:
        df23 = pd.read_excel("Qualar2023.xlsx")
        df23.columns = df23.columns.str.strip()
        if 'Local' in df23.columns: df23 = df23.drop('Local', axis=1)
        df23 = df23.rename(columns={'Data-Hora': 'Data'})
        df23['Data'] = pd.to_datetime(df23['Data'], dayfirst=True, errors='coerce')
        df23['Ano'] = 2023
        for col in cols_num:
            if col in df23.columns: df23[col] = pd.to_numeric(df23[col], errors='coerce')
        data["ar_2023"] = df23
    except:
        pass # Ignora se falhar, para focar no fluxo principal

    # --- CARREGAR METEOROLOGIA ---
    try:
        df_meteo = pd.read_csv("dataset_meteorologico_portugal.csv")
        df_meteo['date'] = pd.to_datetime(df_meteo['date'])
        df_meteo['distrito'] = df_meteo['distrito'].str.strip().str.title()
        data["meteo"] = df_meteo
    except:
        pass

    return data

# Executar carregamento
dados_raw = load_and_process_data()
if not dados_raw:
    st.stop()

df25 = dados_raw['ar_2025']
df23 = dados_raw.get('ar_2023')
df_meteo = dados_raw.get('meteo')

# --- 2. PROCESSAMENTO (Réplica da tua lógica de Classes e Merge) ---

# Filtrar Distritos Desejados (Lógica do código)
distritos_desejados = ['Aveiro', 'Lisboa', 'Açores', 'Setúbal', 'Leiria', 'Madeira', 'Santarém']
df_ar = df25[df25['Distrito'].isin(distritos_desejados)].copy()

# Definição de Intervalos e Classificação (Cópia exata da tua função)
intervalos = {
    "PM10": [(101, 1200), (51, 100), (36, 50), (21, 35), (0, 20)],
    "PM2.5": [(51, 800), (26, 50), (21, 25), (11, 20), (0, 10)],
    "NO2": [(401, 1000), (201, 400), (101, 200), (41, 100), (0, 40)],
    "O3": [(241, 600), (181, 240), (101, 180), (81, 100), (0, 80)],
    "SO2": [(501, 1250), (351, 500), (201, 350), (101, 200), (0, 100)]
}

def classificar_proximo(valor, intervalos):
    distancias = []
    for i, (minimo, maximo) in enumerate(intervalos):
        centro = (minimo + maximo) / 2
        distancias.append((abs(valor - centro), i + 1)) 
    return min(distancias)[1]

for poluente in intervalos:
    # Aplicar em 2025
    df_ar[f'{poluente}_classe'] = df_ar[poluente].apply(lambda x: classificar_proximo(x, intervalos[poluente]))
    # Aplicar em 2023 se existir
    if df23 is not None and poluente in df23.columns:
        df23[f'{poluente}_classe'] = df23[poluente].apply(lambda x: classificar_proximo(x, intervalos[poluente]))

# Calcular Media_Classe
colunas_classes = [c for c in df_ar.columns if c.endswith('_classe')]
df_ar['Media_Classe'] = df_ar[colunas_classes].mean(axis=1)

if df23 is not None:
    # Preparar 2023 (Filtros e Media Classe)
    df23_clean = df23[df23['Distrito'].isin(distritos_desejados)].copy()
    colunas_classes_23 = [c for c in df23_clean.columns if c.endswith('_classe')]
    df23_clean['Media_Classe'] = df23_clean[colunas_classes_23].mean(axis=1)

poluentesclean = ['NO2', 'O3', 'PM2.5', 'PM10', 'SO2']

# ==============================================================================
# VISUALIZAÇÕES (EXATAMENTE COMO PEDIDO)
# ==============================================================================

st.header("1. Boxplots Componentes Diários 2025")
# Lógica do código: Agrupar por data e boxplot
df_por_dia = df_ar.groupby("Data")[poluentesclean].mean().reset_index()

fig1, axes = plt.subplots(nrows=1, ncols=len(poluentesclean), figsize=(18, 5))
for i, col in enumerate(poluentesclean):
    sns.boxplot(y=df_por_dia[col], ax=axes[i], color="skyblue")
    axes[i].set_title(f'Boxplot de {col} em 2025')
    axes[i].set_ylabel("")
plt.tight_layout()
st.pyplot(fig1)

# ------------------------------------------------------------------------------

if df23 is not None:
    st.header("2. Comparação Distribuição 2023 vs 2025")
    
    # Preparar dados para comparação
    df23_periodo = df23_clean.copy()
    df25_periodo = df_ar.copy()
    
    # Match datas (Simplificado para o streamlit rodar rápido)
    datas_2025 = df25_periodo["Data"].dt.strftime("%m-%d").unique()
    df23_periodo = df23_periodo[df23_periodo["Data"].dt.strftime("%m-%d").isin(datas_2025)]
    
    df_comparacao = pd.concat([df23_periodo, df25_periodo], ignore_index=True)
    
    fig2, axes = plt.subplots(2, 3, figsize=(15,8))
    axes = axes.flatten()
    for i, p in enumerate(poluentesclean):
        # Usar seaborn para facilitar o 'by' do pandas plot
        sns.boxplot(data=df_comparacao, x='Ano', y=p, ax=axes[i], palette="Set2")
        axes[i].set_title(p)
    
    # Limpar eixos vazios
    if len(poluentesclean) < len(axes):
        for j in range(len(poluentesclean), len(axes)):
            fig2.delaxes(axes[j])
            
    plt.suptitle("Distribuição dos poluentes — comparação 2023 vs 2025")
    plt.tight_layout()
    st.pyplot(fig2)

    # ------------------------------------------------------------------------------
    st.header("3. Variação Percentual por Distrito")
    
    df23_avg = df23_periodo.groupby("Distrito")[poluentesclean].mean().reset_index()
    df25_avg = df25_periodo.groupby("Distrito")[poluentesclean].mean().reset_index()
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
        
    if len(poluentesclean) < len(axes):
        for j in range(len(poluentesclean), len(axes)):
            fig3.delaxes(axes[j])
            
    plt.tight_layout()
    st.pyplot(fig3)

    # ------------------------------------------------------------------------------
    st.header("4. Comparação Média Geral da Classe")
    media_geral = pd.DataFrame({
        'Ano': ['2023', '2025'],
        'Media_Classe': [df23_periodo['Media_Classe'].mean(), df25_periodo['Media_Classe'].mean()]
    })
    
    fig4 = plt.figure(figsize=(6,5))
    ax = sns.barplot(data=media_geral, x='Ano', y='Media_Classe', palette=['#FFA500', '#1F77B4'])
    for p in ax.patches:
        ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom')
    st.pyplot(fig4)

# ------------------------------------------------------------------------------

if df_meteo is not None:
    st.header("5. Correlações (Meteo vs Qualidade Ar)")
    
    # Preparar merge (por dia e distrito)
    df_ar["Dia"] = df_ar["Data"].dt.date
    df_meteo["Dia"] = df_meteo["date"].dt.date 
    
    df_combinado = pd.merge(df_ar, df_meteo, left_on=["Distrito", "Dia"], right_on=["distrito", "Dia"], how="left")
    
    # Heatmap 1: Meteo vs Media Classe
    cols_meteo = ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_80m", "Media_Classe"]
    df_corr1 = df_combinado[cols_meteo].corr()
    
    fig5 = plt.figure(figsize=(8, 6))
    sns.heatmap(df_corr1, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(fig5)
    
    # Heatmap 2: Meteo vs Poluentes
    st.subheader("Correlação Meteo vs Componentes")
    cols_full = ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_80m"] + poluentesclean
    df_corr2 = df_combinado[cols_full].corr()
    # Filtrar apenas Meteo x Poluentes
    corr_sub = df_corr2.loc[[c for c in cols_full if c in df_corr2.index and c not in poluentesclean], 
                            [c for c in poluentesclean if c in df_corr2.columns]]
    
    fig6 = plt.figure(figsize=(10, 6))
    sns.heatmap(corr_sub, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot(fig6)

# ==============================================================================
# MACHINE LEARNING - SVR AUTOREGRESSIVO (Para Lisboa, conforme o código)
# ==============================================================================

st.header("6. Machine Learning: SVR Autoregressivo")
st.markdown("Previsão para o distrito de **Lisboa** (Lógica do código: `dfL = df_Model[df_Model['distrito'] == 'Lisboa']`)")

# Preparar dados para o SVR (exatamente como no snippet final)
df_class = df_ar[df_ar['Distrito'] == 'Lisboa'].sort_values('Data').copy()

if not df_class.empty:
    # Criar Lags (1 a 7)
    for lag in range(1, 8):
        df_class[f"lag{lag}"] = df_class["Media_Classe"].shift(lag)
    
    df_class = df_class.dropna()
    
    # Definir X e y
    X = df_class[[f"lag{i}" for i in range(1, 8)]]
    y = df_class["Media_Classe"]
    
    # Treinar Modelo (Rápido o suficiente para fazer live)
    try:
        model_ar = SVR(C=10, epsilon=0.1, gamma=0.01)
        model_ar.fit(X, y)
        y_pred = model_ar.predict(X)
        
        # Métricas
        c1, c2, c3 = st.columns(3)
        c1.metric("MAE", f"{mean_absolute_error(y, y_pred):.4f}")
        c2.metric("RMSE", f"{np.sqrt(mean_squared_error(y, y_pred)):.4f}")
        c3.metric("R2 Score", f"{r2_score(y, y_pred):.4f}")
        
        # Gráfico Plotly (O Grand Finale do teu código)
        dates = df_class["Data"]
        
        fig_svr = go.Figure()
        fig_svr.add_trace(go.Scatter(x=dates, y=y, mode="lines", name="Real", line=dict(color="navy")))
        fig_svr.add_trace(go.Scatter(x=dates, y=y_pred, mode="lines", name="Previsto (SVR)", line=dict(color="red")))
        
        fig_svr.update_layout(title="SVR Autoregressivo - Real vs Previsto", xaxis_title="Data", yaxis_title="Media_Classe", template="plotly_white")
        st.plotly_chart(fig_svr, use_container_width=True)
        
    except Exception as e:
        st.error(f"Erro ao treinar SVR: {e}")
else:
    st.warning("Sem dados suficientes de Lisboa para gerar o modelo SVR.")
