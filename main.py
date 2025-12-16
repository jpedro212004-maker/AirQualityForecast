import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="Qualidade do Ar Portugal",
    page_icon="🇵🇹",
    layout="wide"
)

# --- FUNÇÃO DE CARREGAMENTO DE DADOS (COM CACHE) ---
@st.cache_data
def load_data():
    data = {}
    
    # Lista de colunas que TÊM de ser numéricas
    # Isto força "N.D.", "-" ou erros a virarem NaN para não bloquear os cálculos
    colunas_numericas = ['NO2', 'O3', 'PM10', 'PM2.5', 'SO2', 'CO', 'C6H6']

    # 1. Carregar Qualidade do Ar 2025
    try:
        df25 = pd.read_excel("QualidadeAr2.xlsx")
        
        # Renomear coluna Distrito se necessário
        if 'Coluna1' in df25.columns:
            df25 = df25.rename(columns={"Coluna1": "Distrito"})
        
        # Converter Data
        df25['Data'] = pd.to_datetime(df25['Data'], dayfirst=True, errors='coerce')
        df25['Ano'] = 2025

        # LIMPEZA CRÍTICA: Converter colunas de poluentes para números
        for col in colunas_numericas:
            if col in df25.columns:
                df25[col] = pd.to_numeric(df25[col], errors='coerce')

        data["ar_2025"] = df25
    except Exception as e:
        st.error(f"Erro ao carregar 2025: {e}")

    # 2. Carregar Qualidade do Ar 2023
    try:
        df23 = pd.read_excel("Qualar2023.xlsx")
        
        # Limpezas básicas de colunas
        df23.columns = df23.columns.str.strip()
        if 'Local' in df23.columns:
             df23 = df23.drop('Local', axis=1)
             
        df23 = df23.rename(columns={'Data-Hora': 'Data'})
        df23['Data'] = pd.to_datetime(df23['Data'], dayfirst=True, errors='coerce')
        df23['Ano'] = 2023
        
        # LIMPEZA CRÍTICA: Converter colunas de poluentes para números
        for col in colunas_numericas:
            if col in df23.columns:
                df23[col] = pd.to_numeric(df23[col], errors='coerce')

        data["ar_2023"] = df23
    except Exception as e:
        st.warning(f"Aviso: Dados de 2023 não encontrados ou com erro ({e})")

    # 3. Carregar Meteorologia
    try:
        df_meteo = pd.read_csv("dataset_meteorologico_portugal.csv")
        df_meteo['date'] = pd.to_datetime(df_meteo['date'])
        
        # Uniformizar nomes de distritos no dataset de meteorologia
        if 'distrito' in df_meteo.columns:
            df_meteo['distrito_clean'] = df_meteo['distrito'].str.strip().str.title()
            
        data["meteo"] = df_meteo
    except Exception as e:
        st.warning(f"Aviso: Dados meteorológicos não encontrados ({e})")
        
    return data

# --- EXECUÇÃO DO CARREGAMENTO ---
dados = load_data()
df2025 = dados.get("ar_2025")
df2023 = dados.get("ar_2023")
df_meteo = dados.get("meteo")

# --- CABEÇALHO ---
st.title("🇵🇹 Monitorização da Qualidade do Ar em Portugal")
st.markdown("""
Esta aplicação visualiza dados de qualidade do ar e meteorologia, permitindo comparar a evolução
entre **2023 e 2025**.  
*Desenvolvido no âmbito do projeto de Ciência de Dados.*
""")
st.divider()

# --- BARRA LATERAL (FILTROS) ---
st.sidebar.header("Filtros")

# Obter lista de distritos
if df2025 is not None:
    lista_distritos = sorted(df2025['Distrito'].astype(str).unique())
    distrito_selecionado = st.sidebar.selectbox("Escolha o Distrito:", lista_distritos)
else:
    st.error("Não foi possível carregar os dados de 2025. Verifique os ficheiros no GitHub.")
    st.stop()

# --- FILTRAGEM DOS DATAFRAMES ---
# Filtrar 2025
df2025_filtrado = df2025[df2025['Distrito'] == distrito_selecionado].sort_values('Data')

# Filtrar 2023
if df2023 is not None:
    df2023_filtrado = df2023[df2023['Distrito'] == distrito_selecionado].sort_values('Data')
else:
    df2023_filtrado = pd.DataFrame()

# Filtrar Meteo
if df_meteo is not None:
    df_meteo_filtrado = df_meteo[df_meteo['distrito_clean'] == distrito_selecionado].sort_values('date')
else:
    df_meteo_filtrado = pd.DataFrame()


# --- TABS DE CONTEÚDO ---
tab1, tab2, tab3, tab4 = st.tabs(["📈 Evolução Temporal", "🆚 Comparação 23/25", "🌤️ Meteorologia", "🤖 Modelação (Resultados)"])

with tab1:
    st.header(f"Qualidade do Ar em 2025: {distrito_selecionado}")
    
    colunas_poluentes = ['NO2', 'O3', 'PM10', 'PM2.5', 'SO2']
    # Verificar quais colunas existem no DF filtrado
    cols_existentes = [c for c in colunas_poluentes if c in df2025_filtrado.columns]
    
    if cols_existentes:
        poluente = st.selectbox("Selecione o Poluente:", cols_existentes)
        
        # Gráfico de Linhas Interativo
        fig = px.line(df2025_filtrado, x='Data', y=poluente, 
                      title=f"Evolução Diária de {poluente} (2025)",
                      markers=True, template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
        
        # Métricas
        # O dropna() garante que calculamos média apenas de valores válidos
        series_clean = df2025_filtrado[poluente].dropna()
        if not series_clean.empty:
            avg_val = series_clean.mean()
            max_val = series_clean.max()
            
            col1, col2 = st.columns(2)
            col1.metric("Média 2025", f"{avg_val:.2f} µg/m³")
            col2.metric("Máximo Registado", f"{max_val:.2f} µg/m³")
        else:
            st.info("Sem dados válidos para este poluente neste distrito.")
    else:
        st.warning("Colunas de poluentes não encontradas.")

with tab2:
    st.header("Comparação 2023 vs 2025")
    
    if not df2023_filtrado.empty and not df2025_filtrado.empty:
        # Calcular médias anuais para o gráfico
        # numeric_only=True evita erro se houver colunas de texto perdidas
        media23 = df2023_filtrado[cols_existentes].mean(numeric_only=True).reset_index()
        media23.columns = ['Poluente', 'Valor']
        media23['Ano'] = '2023'
        
        media25 = df2025_filtrado[cols_existentes].mean(numeric_only=True).reset_index()
        media25.columns = ['Poluente', 'Valor']
        media25['Ano'] = '2025'
        
        df_comp = pd.concat([media23, media25])
        
        if not df_comp.empty:
            # Gráfico de Barras Agrupadas
            fig_comp = px.bar(df_comp, x='Poluente', y='Valor', color='Ano', barmode='group',
                              title=f"Média de Poluentes: {distrito_selecionado}",
                              color_discrete_sequence=['#FFA500', '#1F77B4']) # Laranja e Azul
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.warning("Não há dados suficientes para gerar o gráfico de comparação.")
    else:
        st.info("Dados de 2023 ou 2025 incompletos para comparação direta.")

with tab3:
    st.header("Condições Meteorológicas (2025)")
    
    if not df_meteo_filtrado.empty:
        var_meteo = st.radio("Variável:", ["temperature_2m", "rain", "wind_speed_80m", "relative_humidity_2m"], horizontal=True)
        
        fig_meteo = px.area(df_meteo_filtrado, x='date', y=var_meteo,
                            title=f"Evolução de {var_meteo}",
                            color_discrete_sequence=['#2ca02c'])
        st.plotly_chart(fig_meteo, use_container_width=True)
        
        # Mapa de Calor Correlação (Meteo vs Poluentes)
        st.subheader("Correlação: Meteo vs Poluentes")
        
        # Juntar dados por data para correlação
        # Resample para média diária
        df_meteo_dia = df_meteo_filtrado.set_index('date').resample('D').mean(numeric_only=True)
        df_ar_dia = df2025_filtrado.set_index('Data')[cols_existentes].resample('D').mean(numeric_only=True)
        
        # Juntar os dois dataframes pelo índice (Data)
        df_corr_full = pd.concat([df_meteo_dia[['temperature_2m', 'rain', 'wind_speed_80m']], df_ar_dia], axis=1)
        
        # Calcular correlação
        df_corr = df_corr_full.corr()
        
        fig_corr = px.imshow(df_corr, text_auto=True, color_continuous_scale='RdBu_r', title="Matriz de Correlação", aspectRatio=1)
        st.plotly_chart(fig_corr, use_container_width=True)
        
    else:
        st.info("Dados meteorológicos indisponíveis para este distrito.")

with tab4:
    st.header("🔬 Resultados de Machine Learning")
    st.markdown("""
    Nesta secção apresentamos os resultados dos modelos treinados offline para prever a qualidade do ar.
    
    **Objetivo:** Prever a concentração de poluentes com base na meteorologia e histórico.
    """)
    
    st.subheader("🏆 Melhores Modelos por Poluente")
    
    # Dados baseados na tua análise no notebook
    resultados = {
        "Poluente": ["NO2", "O3", "PM10", "PM2.5", "SO2"],
        "Melhor Modelo": ["MLP (Rede Neural)", "LightGBM", "LightGBM", "RandomForest", "LightGBM"],
        "MAE (Erro Médio)": ["13.42", "18.29", "3.11", "7.03", "0.43"],
        "R² Score": ["0.30", "-0.44", "0.31", "-0.49", "0.64"]
    }
    st.table(pd.DataFrame(resultados))
    
    st.info("""
    **Conclusões Principais:**
    * ✅ O **SO2** obteve a melhor previsão (R² 0.64), indicando forte relação com as variáveis disponíveis.
    * ⚠️ O **O3 (Ozono)** e **PM2.5** mostraram-se muito difíceis de prever apenas com meteorologia (R² negativo).
    * 🧠 As **Redes Neurais** funcionaram bem para o NO2.
    """)
