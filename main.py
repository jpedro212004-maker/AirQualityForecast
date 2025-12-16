# %%
#imports
import os
import time
import random
import math
import csv
import requests
import requests_cache
import openmeteo_requests

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.subplots as sp
import plotly.graph_objects as go

from scipy.stats import boxcox

from retry_requests import retry

from datetime import datetime

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

# Scikit-learn
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Regressão
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor

# Classificação
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier

# Modelos de séries temporais
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

# Boosting libraries
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

import plotly.io as pio

# %% [markdown]
# # Introdução
# 

# %% [markdown]
# ## Extração de Dados Necessários

# %% [markdown]
# ### Extração dos Dados de Meteorologia

# %%
#Configuração da API
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

#Coordenadas dos distritos
distritos = {
    "Aveiro": (40.6405, -8.6538),
    "Beja": (38.0151, -7.8632),
    "Braga": (41.5503, -8.4201),
    "Bragança": (41.8060, -6.7567),
    "Castelo Branco": (39.8239, -7.4931),
    "Coimbra": (40.2110, -8.4292),
    "Évora": (38.5667, -7.9000),
    "Faro": (37.0194, -7.9322),
    "Guarda": (40.5373, -7.2658),
    "Leiria": (39.7436, -8.8071),
    "Lisboa": (38.7169, -9.1399),
    "Portalegre": (39.2938, -7.4312),
    "Porto": (41.1496, -8.6109),
    "Santarém": (39.2362, -8.6861),
    "Setúbal": (38.5244, -8.8882),
    "Viana do Castelo": (41.6918, -8.8344),
    "Vila Real": (41.3000, -7.7441),
    "Viseu": (40.6610, -7.9097),
    "Açores": (37.7392, -25.6687),
    "Madeira": (32.6669, -16.9241)
}

#Parâmetros comuns da API
url = "https://api.open-meteo.com/v1/forecast"
params_base = {
    "daily": ["temperature_2m_max", "temperature_2m_min", "uv_index_max"],
    "hourly": ["temperature_2m", "relative_humidity_2m", "rain",
               "temperature_80m", "wind_speed_80m", "wind_direction_80m"],
    "past_days": 93,
    "forecast_days": 3,
}

#Lista onde guardamos todos os distritos
tabela_geral = []

#Loop pelos distritos
for nome, (lat, lon) in distritos.items():
    print(f"\n--- Obtendo dados para {nome} ({lat}, {lon}) ---")
    params = params_base | {"latitude": lat, "longitude": lon}

    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]

        #DADOS HORÁRIOS
        hourly = response.Hourly()
        hourly_index = pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left"
        )
        hourly_vars = ["temperature_2m", "relative_humidity_2m", "rain",
                       "temperature_80m", "wind_speed_80m", "wind_direction_80m"]
        hourly_data = {"date": hourly_index}
        for i, var in enumerate(hourly_vars):
            try:
                vals = hourly.Variables(i).ValuesAsNumpy()
                if len(vals) != len(hourly_index):
                    vals = np.full(len(hourly_index), np.nan)
                hourly_data[var] = vals
            except Exception:
                hourly_data[var] = np.full(len(hourly_index), np.nan)
        hourly_df = pd.DataFrame(hourly_data)

        #DADOS DIÁRIOS
        daily = response.Daily()
        daily_index = pd.date_range(
            start=pd.to_datetime(daily.Time(), unit="s", utc=True),
            end=pd.to_datetime(daily.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=daily.Interval()),
            inclusive="left"
        )
        daily_vars = ["temperature_2m_max", "temperature_2m_min", "uv_index_max"]
        daily_data = {"date": daily_index}
        for i, var in enumerate(daily_vars):
            try:
                vals = daily.Variables(i).ValuesAsNumpy()
                if len(vals) != len(daily_index):
                    vals = np.full(len(daily_index), np.nan)
                daily_data[var] = vals
            except Exception:
                daily_data[var] = np.full(len(daily_index), np.nan)
        daily_df = pd.DataFrame(daily_data)

        #JUNÇÃO (repete dados diários para cada hora do mesmo dia)
        hourly_df["day"] = pd.to_datetime(hourly_df["date"]).dt.floor("D")
        daily_df["day"] = pd.to_datetime(daily_df["date"]).dt.floor("D")

        merged = hourly_df.merge(daily_df.drop(columns=["date"]), on="day", how="left")
        merged.drop(columns=["day"], inplace=True)

        #Adiciona o distrito
        merged.insert(0, "distrito", nome)

        #Adiciona à tabela geral
        tabela_geral.append(merged)

        print(f"✅ {nome}: {len(merged)} linhas adicionadas.")

    except Exception as e:
        print(f"❌ Erro ao obter dados para {nome}: {e}")

#Combina tudo num único dataset
df_meteo = pd.concat(tabela_geral, ignore_index=True)

#Converte a data para timezone local (opcional)
df_meteo["date"] = pd.to_datetime(df_meteo["date"]).dt.tz_convert("Europe/Lisbon")

#Exporta como CSV único
os.makedirs("dados_meteo", exist_ok=True)
df_meteo.to_csv("dados_meteo/dataset_meteorologico_portugal.csv", index=False)

print("\n✅ Dataset final criado com sucesso!")
print(f"Linhas totais: {len(df_meteo)}")
print("Amostra:")
#df_meteo=df_meteo.columns[["Distrito", "Data", "temperature_2m" , "relative_humidity_2m","rain","temperature_80m","wind_speed_80m","wind_direction_80m","temperature_2m_max","temperature_2m_min","uv_index_max"]]


df_meteo.to_csv("dataset_meteorologico_portugal.csv", index=False)

# %%
df_meteo=pd.read_csv("dataset_meteorologico_portugal.csv")
df_meteo["date"] = pd.to_datetime(df_meteo["date"])

df_meteo

# %% [markdown]
# ### Extração da Qualidade do Ar para o Próprio dia

# %%

TOKEN = "dbff977abd2c7f76045d493dc54f67e125a633c3"  # Substitr pelo seu token WAQI
HISTORIC_FILE = "QualidadeAr2.xlsx" # Nome do dataset histórico no formato Excel

# Lista simplificada de concelhos com coordenadas
distritos2 = [
    ("Aveiro", 40.6405, -8.6538), ("Beja", 38.0151, -7.8632),
    ("Braga", 41.5503, -8.4201), ("Bragança", 41.8060, -6.7567),
    ("Castelo Branco", 39.8239, -7.4931), ("Coimbra", 40.2110, -8.4292),
    ("Évora", 38.5667, -7.9000), ("Faro", 37.0194, -7.9322),
    ("Guarda", 40.5373, -7.2658), ("Leiria", 39.7436, -8.8071),
    ("Lisboa", 38.7169, -9.1399), ("Portalegre", 39.2938, -7.4312),
    ("Porto", 41.1496, -8.6109), ("Santarém", 39.2362, -8.6861),
    ("Setúbal", 38.5244, -8.8882), ("Viana do Castelo", 41.6918, -8.8344),
    ("Vila Real", 41.3000, -7.7441), ("Viseu", 40.6610, -7.9097),
    ("Açores", 37.7392, -25.6687), ("Madeira", 32.6669, -16.9241)
]


# Variáveis de controle
TODAY_DATE = datetime.now().date()
API_CALL_NEEDED_AR = True
latest_date_in_df = None

# --- LÓGICA DE VERIFICAÇÃO ---
if os.path.exists(HISTORIC_FILE):
    try:
        df_historico = pd.read_excel(HISTORIC_FILE)
        df_historico['Data'] = pd.to_datetime(df_historico['Data'], errors='coerce')
        latest_date_in_df = df_historico['Data'].max()
        
        # Se a data mais recente no DF for igual à data de hoje, pula a API
        if latest_date_in_df is not pd.NaT and latest_date_in_df.date() == TODAY_DATE:
            print(f"⚠️ O dataset de Qualidade do Ar já tem dados de hoje ({latest_date_in_df.strftime('%Y-%m-%d %H:%M:%S')}). Pulando chamada da API.")
            API_CALL_NEEDED_AR = False
        elif latest_date_in_df is not pd.NaT:
            print(f"Último registro no dataset: {latest_date_in_df.strftime('%Y-%m-%d %H:%M:%S')}. Buscando novos dados.")
        
    except Exception as e:
        print(f"❌ ERRO ao ler o arquivo XLSX: {e}. Prosseguindo com a coleta da API.")
        df_historico = None
else:
    print(f"Arquivo {HISTORIC_FILE} não encontrado. Será criado um novo.")
    df_historico = None


# --- FUNÇÃO PARA OBTER DADOS POR COORDENADAS ---
def obter_dados_geo(nome, lat, lon):
    url = f"https://api.waqi.info/feed/geo:{lat};{lon}/?token={TOKEN}"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro de requisição para {nome}: {e}")
        return None

    if data["status"] == "ok":
        d = data["data"]
        return {
            "Distrito": nome,
            "O3": d.get("iaqi", {}).get("o3", {}).get("v"),
            "NO2": d.get("iaqi", {}).get("no2", {}).get("v"),
            "CO": d.get("iaqi", {}).get("co", {}).get("v"),
            "SO2": d.get("iaqi", {}).get("so2", {}).get("v"),
            "PM10": d.get("iaqi", {}).get("pm10", {}).get("v"),
            "PM2.5": d.get("iaqi", {}).get("pm25", {}).get("v"),
            "C6H6": None, # Benceno não fornecido
            "Data": d["time"]["s"], 
            "aqi_geral": d.get("aqi"), # Apenas para conferência
        }
    else:
        print(f"⚠️ Sem dados disponíveis ou token inválido para {nome}. Status: {data['status']}")
        return None

# --- EXECUÇÃO E COLETA DE DADOS RECENTES ---
dados_recentes = []

if API_CALL_NEEDED_AR:
    print(f"\n🔍 A obter dados de qualidade do ar (hoje) para {len(distritos2)} distritos...\n")
    for i, (nome, lat, lon) in enumerate(distritos2, start=1):
        print(i, nome, lat, lon)
        info = obter_dados_geo(nome, lat, lon)
        if info:
            dados_recentes.append(info)
        time.sleep(1.2) # Intervalo para evitar bloqueio da API

    print("\n--- Processamento e Fusão ---")

    if not dados_recentes:
        print("❌ Nenhum dado recente foi obtido. O dataset histórico não será alterado.")
        df_atualizado = df_historico
    else:
        # 1. Criar o DataFrame de Novos Dados
        df_novos = pd.DataFrame(dados_recentes)

        # 2. AJUSTE NA FORMATAÇÃO DA COLUNA 'DATA' (TimeZone-Naive)
        df_novos["Data"] = pd.to_datetime(df_novos["Data"]) 

        # 3. Adicionar Colunas Faltantes e Garantir a Ordem
        df_novos["Tipo"] = np.nan
        df_novos["Zona"] = np.nan
        df_novos["Período"] = np.nan
        df_novos["C6H6"] = np.nan

        # Remove colunas desnecessárias
        df_novos.drop(columns=["aqi_geral"], errors='ignore', inplace=True) 

        COLUMNS_ORDER = ['Tipo', 'Distrito', 'Zona', 'Período', 'O3', 'NO2', 'CO', 'SO2', 'PM10', 'PM2.5', 'C6H6', 'Data']
        df_novos = df_novos.reindex(columns=COLUMNS_ORDER)

        # 4. Carregar e Fusão (Atualização)
        if df_historico is not None:
            # Filtrar Duplicados (Remove o registro se a data for igual ou anterior ao máximo)
            df_novos_filtrados = df_novos[df_novos['Data'] > latest_date_in_df]
            
            # Concatenar (apenas os NOVOS, sem duplicados)
            df_atualizado = pd.concat([df_historico, df_novos_filtrados], ignore_index=True)
            
            linhas_adicionadas = len(df_novos_filtrados)
            print(f"\n✅ Dataset Histórico (XLSX) atualizado. Linhas adicionadas: {linhas_adicionadas}")
        else:
            # Se o histórico não existe, criamos o arquivo apenas com os dados de hoje.
            df_atualizado = df_novos
            print("\n✅ Dataset Histórico (XLSX) criado a partir dos dados de hoje.")

    # 5. Salvar o Dataset Único
    if df_atualizado is not None:
        try:
            df_atualizado.to_excel(HISTORIC_FILE, index=False)
            print(f"Linhas totais no dataset atualizado ({HISTORIC_FILE}): {len(df_atualizado)}")
        except PermissionError:
             print(f"\n❌ ERRO: Permissão negada ao acessar {HISTORIC_FILE}. Feche o arquivo no Excel e tente novamente.")
             df_atualizado = df_historico # Volta ao anterior se falhar ao salvar
        except Exception as e:
             print(f"\n❌ ERRO ao salvar o arquivo XLSX: {e}")
             df_atualizado = df_historico # Volta ao anterior se falhar ao salvar
else:
    print("✅ Pulando a chamada da API WAQI. Dataset já atualizado para hoje.")
    df_atualizado = df_historico

dfqualidadear = df_atualizado.copy()
print("\nFim da etapa de atualização da Qualidade do Ar.")

# %% [markdown]
# ### Extração dos Dados da Qualidade do Ar 2025

# %%
dfqualidadear=pd.read_excel("QualidadeAr2.xlsx")
dfqualidadear=dfqualidadear.rename(columns={"Coluna1":"Distrito"})
dfqualidadear.columns
dfqualidadear=dfqualidadear.drop(["Tipo", "Zona"], axis=1)
dfqualidadear

# %%
# 1. Converter 'Data' para datetime
dfqualidadear['Data'] = pd.to_datetime(dfqualidadear['Data'], dayfirst=True)

# 2. Extrair semana e ano
dfqualidadear['Semana'] = dfqualidadear['Data'].dt.isocalendar().week
dfqualidadear['Ano'] = dfqualidadear['Data'].dt.year

# 3. Substituir 'N.D.' por NaN e converter colunas para numérico
poluentes = ['C6H6', 'CO', 'NO2', 'O3', 'PM2.5', 'PM10', 'SO2']
for p in poluentes:
    dfqualidadear[p] = pd.to_numeric(dfqualidadear[p], errors='coerce')

# 4. Calcular médias semanais por distrito
df_medias = dfqualidadear.groupby(['Distrito', 'Ano', 'Semana'])[poluentes].mean().reset_index()

# 5. Juntar ao DataFrame original
dfqualidadear = dfqualidadear.merge(df_medias, on=['Distrito', 'Ano', 'Semana'], suffixes=('', '_media'))

# 6. Preencher NaNs com as médias semanais
for p in poluentes:
    dfqualidadear[p] = dfqualidadear[p].fillna(dfqualidadear[f'{p}_media'])

# %%
# Lista dos poluentes que queres verificar
poluentes = ['C6H6', 'CO', 'NO2', 'O3', 'PM2.5', 'PM10', 'SO2']

# Agrupa por distrito e conta os NaNs por poluente
nan_por_distrito = dfqualidadear.groupby('Distrito')[poluentes].apply(lambda x: x.isna().sum())
nan_por_distrito

# %%
df_mediaar = dfqualidadear.groupby(["Data","Distrito","Semana","Ano"])[poluentes].mean().reset_index()
df_mediaar

# %%
#Criamos classes para cada poluente, onde 1-Mau 2-Fraco 3-Médio 4-Bom 5-Muito bom

def classificar(valor, limites):
    for i, (minimo, maximo) in enumerate(limites):
        if minimo <= valor <= maximo:
            return i + 1  # Agora 1 = Mau, 5 = Muito Bom
    return None  # Se estiver fora dos limites

# Define os intervalos por poluente (na ordem correta: Mau → Muito Bom)
intervalos = {
    "PM10": [(101, 1200), (51, 100), (36, 50), (21, 35), (0, 20)],
    "PM2.5": [(51, 800), (26, 50), (21, 25), (11, 20), (0, 10)],
    "NO2": [(401, 1000), (201, 400), (101, 200), (41, 100), (0, 40)],
    "O3": [(241, 600), (181, 240), (101, 180), (81, 100), (0, 80)],
    "SO2": [(501, 1250), (351, 500), (201, 350), (101, 200), (0, 100)]
}

# Aplica a classificação a cada poluente

def classificar_proximo(valor, intervalos):
    distancias = []
    for i, (minimo, maximo) in enumerate(intervalos):
        centro = (minimo + maximo) / 2
        distancias.append((abs(valor - centro), i + 1))  # i+1 para manter classe 1 a 5
    # Retorna a classe com menor distância
    return min(distancias)[1]

for poluente in intervalos:
    df_mediaar[f'{poluente}_classe'] = df_mediaar[poluente].apply(lambda x: classificar_proximo(x, intervalos[poluente]))


df_mediaar   

# %%
print(df_mediaar.isna().sum())

# %%
df_ar =df_mediaar.drop(['C6H6', 'CO'], axis=1)
distritos_desejados = ['Aveiro', 'Lisboa', 'Açores', 'Setúbal', 'Leiria', 'Madeira', 'Santarém']
df_ar = df_ar[df_ar['Distrito'].isin(distritos_desejados)]


df_ar.isna().any().any()
# Identifica automaticamente as colunas de classe
colunas_classes = [c for c in df_ar.columns if c.endswith('_classe')]

# Calcula a média das classes para cada linha
df_ar['Media_Classe'] = df_ar[colunas_classes].mean(axis=1)

df_ar

# %%
print(df_ar.isna().sum())

# %% [markdown]
# ### Extração dos Dados da Qualidade do Ar 2023

# %%
dfqualidadear2023 = pd.read_excel("Qualar2023.xlsx")

dfqualidadear2023.columns = dfqualidadear2023.columns.str.strip()
dfqualidadear2023=dfqualidadear2023.drop('Local', axis=1)
dfqualidadear2023

# %%
# Lista dos poluentes que queres verificar
poluentes = ['C6H6', 'CO', 'NO2', 'O3', 'PM2.5', 'PM10', 'SO2']

# Agrupa por distrito e conta os NaNs por poluente
nan_por_distrito2023 = dfqualidadear2023.groupby('Distrito')[poluentes].apply(lambda x: x.isna().sum())
nan_por_distrito2023

# %%
#Juntar a partir da média semanal
dfqualidadear2023['Data-Hora'] = pd.to_datetime(dfqualidadear2023['Data-Hora'])
dfqualidadear2023['Semana'] = dfqualidadear2023['Data-Hora'].dt.isocalendar().week
dfqualidadear2023['Ano'] = dfqualidadear2023['Data-Hora'].dt.year
df_semanalar2023 = dfqualidadear2023.groupby(['Distrito', 'Ano', 'Semana']).mean(numeric_only=True).reset_index()
df_semanalar2023


# %%
dfqualidadear2023 = dfqualidadear2023.merge(
    df_semanalar2023,
    on=['Distrito', 'Ano', 'Semana'],
    suffixes=('', '_media'),
    how='left'
)
# Para cada poluente, preenche NaNs com a média semanal
poluentes = ['C6H6', 'CO', 'NO2', 'O3', 'PM2.5', 'PM10', 'SO2']
for p in poluentes:
    dfqualidadear2023[p] = dfqualidadear2023[p].fillna(dfqualidadear2023[f'{p}_media'])

# %%
print(dfqualidadear2023.isna().sum())

# %%
# Lista dos poluentes que queres verificar
poluentes = ['C6H6', 'CO', 'NO2', 'O3', 'PM2.5', 'PM10', 'SO2']

# Agrupa por distrito e conta os NaNs por poluente
nan_por_distrito2023 = dfqualidadear2023.groupby('Distrito')[poluentes].apply(lambda x: x.isna().sum())
nan_por_distrito2023

# %%
df2023_media = dfqualidadear2023.groupby(["Distrito", "Data-Hora","Semana","Ano"])[poluentes].mean().reset_index()
df2023_media

# %%
#índice da qualidade do ar
#Criamos classes para cada poluente, onde 1-Mau 2-Fraco 3-Médio 4-Bom 5-Muito bom

def classificar(valor, limites):
    for i, (minimo, maximo) in enumerate(limites):
        if minimo <= valor <= maximo:
            return i + 1  # Agora 1 = Mau, 5 = Muito Bom
    return None  # Se estiver fora dos limites

# Define os intervalos por poluente (na ordem correta: Mau → Muito Bom)
intervalos = {
    "PM10": [(101, 1200), (51, 100), (36, 50), (21, 35), (0, 20)],
    "PM2.5": [(51, 800), (26, 50), (21, 25), (11, 20), (0, 10)],
    "NO2": [(401, 1000), (201, 400), (101, 200), (41, 100), (0, 40)],
    "O3": [(241, 600), (181, 240), (101, 180), (81, 100), (0, 80)],
    "SO2": [(501, 1250), (351, 500), (201, 350), (101, 200), (0, 100)]
}

# Aplica a classificação a cada poluente

def classificar_proximo(valor, intervalos):
    distancias = []
    for i, (minimo, maximo) in enumerate(intervalos):
        centro = (minimo + maximo) / 2
        distancias.append((abs(valor - centro), i + 1))  # i+1 para manter classe 1 a 5
    # Retorna a classe com menor distância
    return min(distancias)[1]

for poluente in intervalos:
    df2023_media[f'{poluente}_classe'] = df2023_media[poluente].apply(lambda x: classificar_proximo(x, intervalos[poluente]))

# Identifica automaticamente as colunas de classe
colunas_classes = [c for c in df2023_media.columns if c.endswith('_classe')]

# Calcula a média das classes para cada linha
df2023_media['Media_Classe'] = df2023_media[colunas_classes].mean(axis=1)

df2023_media  

# %%
df2023_mediaclean   =df2023_media  .drop(['C6H6', 'CO'], axis=1)
distritos_desejados = ['Aveiro', 'Lisboa', 'Açores', 'Setúbal', 'Santarém']
df2023_mediaclean  = df2023_mediaclean [df2023_mediaclean ['Distrito'].isin(distritos_desejados)]
poluentesclean = ['NO2', 'O3', 'PM2.5', 'PM10', 'SO2']

df2023_mediaclean .isna().any().any()
df2023_mediaclean['Data-Hora'] = pd.to_datetime(df2023_mediaclean['Data-Hora']).dt.date
df2023_mediaclean.rename(columns={'Data-Hora': 'Data'}, inplace=True)


df2023_mediaclean

# %%
print(df2023_mediaclean.isna().sum())

# %% [markdown]
# ## EDA

# %% [markdown]
# ### Componenetes Diários 2025

# %%

# Converter 'Data-Hora' para datetime e extrair só a data

# Agrupar por data e calcular média dos poluentes
df_por_dia = df_ar.groupby("Data")[poluentesclean].mean().reset_index()

# Criar subplots
fig, axes = plt.subplots(nrows=1, ncols=len(poluentesclean), figsize=(18, 5))

# Gerar boxplots
for i, col in enumerate(poluentesclean):
    sns.boxplot(y=df_por_dia[col], ax=axes[i], color="skyblue")
    axes[i].set_title(f'Boxplot de {col} em 2025')
    axes[i].set_ylabel("")

plt.tight_layout()
plt.show()

# %% [markdown]
# **NO2** apresenta uma distribuição compacta e sem outliers, sugerindo controle eficaz ou estabilidade nas emissões.
# 
# **O3** tem uma mediana alta e vários outliers baixos, o que pode indicar variações sazonais ou condições atmosféricas que reduzem temporariamente os níveis.
# 
# **PM2.5** e **PM10** mostram preocupações com qualidade do ar: embora a mediana seja baixa/moderada, os outliers indicam episódios de poluição intensa.
# 
# **SO2** tem baixa variabilidade, mas os outliers sugerem fontes pontuais de emissão (como atividades industriais).

# %% [markdown]
# ### Comparação da distribuição dos poluentes entre os anos 2023 e 2025

# %%
# Uniformizar nomes
df_ar["Distrito"] = df_ar["Distrito"].str.strip().str.title()
df2023_mediaclean["Distrito"] = df2023_mediaclean["Distrito"].str.strip().str.title()

# Filtrar apenas as mesmas datas de 2025 em 2023
datas_2025 = df_ar["Data"].apply(lambda x: x.strftime("%m-%d")).unique()
df2023_periodo = df2023_mediaclean[
    df2023_mediaclean["Data"].apply(lambda x: x.strftime("%m-%d")).isin(datas_2025)
]

# Manter apenas colunas de interesse
df2023_periodo = df2023_periodo[["Data", "Distrito","PM10_classe","PM2.5_classe","NO2_classe","O3_classe","SO2_classe","Media_Classe"] + poluentesclean]
df2025_periodo = df_ar[["Data", "Distrito","PM10_classe","PM2.5_classe","NO2_classe","O3_classe","SO2_classe","Media_Classe"] + poluentesclean]

print("Datas consideradas:", df2023_periodo["Data"].min(), "a", df2023_periodo["Data"].max())

# Adicionar coluna "Ano"
df2023_periodo["Ano"] = 2023
df2025_periodo["Ano"] = 2025

# Juntar num único DataFrame
df_comparacao = pd.concat([df2023_periodo, df2025_periodo], ignore_index=True)

# Criar boxplots lado a lado
fig, axes = plt.subplots(2, 3, figsize=(15,8))
axes = axes.flatten()

for i, p in enumerate(poluentesclean):
    df_comparacao.boxplot(column=p, by="Ano", ax=axes[i], grid=False, patch_artist=True)
    axes[i].set_title(p)
    axes[i].set_xlabel("Ano")
    axes[i].set_ylabel("Concentração média")

# Remover o último subplot vazio
if len(poluentesclean) < len(axes):
    for j in range(len(poluentesclean), len(axes)):
        fig.delaxes(axes[j])

plt.suptitle("Distribuição dos poluentes — comparação 2023 vs 2025", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# %% [markdown]
# Para uma análise mais aprofundada, podemos calcular a variação percentual por distrito e destacar os distritos com maior aumento ou redução para cada poluente.

# %% [markdown]
# ### Comparação percentual das componentes por distrito entre os anos 2023 e 2025

# %%
# Calcular médias por distrito
df2023_periodo_avg = df2023_periodo.groupby("Distrito")[poluentesclean].mean().reset_index()
df2025_periodo_avg = df2025_periodo.groupby("Distrito")[poluentesclean].mean().reset_index()

# Juntar os dois datasets
comparacao = pd.merge(df2023_periodo_avg, df2025_periodo_avg, on="Distrito", suffixes=("_2023", "_2025"))

# Calcular variação percentual
for p in poluentesclean:
    comparacao[f"{p}_var_percent"] = ((comparacao[f"{p}_2025"] - comparacao[f"{p}_2023"]) / comparacao[f"{p}_2023"]) * 100

# Criar subplots para todos os poluentes
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
axes = axes.flatten()

for i, poluente in enumerate(poluentesclean):
    comparacao_sorted = comparacao.sort_values(f"{poluente}_var_percent", ascending=False)
    axes[i].bar(comparacao_sorted["Distrito"], comparacao_sorted[f"{poluente}_var_percent"])
    axes[i].axhline(0, color="gray", linestyle="--")
    axes[i].set_title(f"{poluente} (2023 → 2025)")
    axes[i].set_ylabel("Variação (%)")
    axes[i].tick_params(axis='x', rotation=45)

# Remover subplot vazio (se houver)
if len(poluentesclean) < len(axes):
    for j in range(len(poluentesclean), len(axes)):
        fig.delaxes(axes[j])

plt.suptitle("Variação Percentual dos Poluentes por Distrito (2023 → 2025)", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # ajustar para não sobrepor o título
plt.show()


# %% [markdown]
# **NO2:** NO₂ está a aumentar em todos os distritos, o que pode refletir maior tráfego, urbanização ou menor regulação.
# 
# **O3:** O aumento de ozono pode estar ligado a reações fotoquímicas em ambientes urbanos. Açores permanece estável, talvez por menor densidade populacional.
# 
# **PM2.5:** Redução em Aveiro e Lisboa, que pode estar ligada a políticas ambientais ou condições meteorológicas favoráveis.
# 
# **PM10:** Tendência positiva geral, exceto Santarém.
# 
# **SO2:** Aumento em todos os distritos, tendo disparado em Aveiro.

# %% [markdown]
# ### Comparação entre a média da qualidade do ar entre 2023 e 2025

# %%
#Preparação dos dados (igual ao anterior)

df2023_periodo['Data'] = pd.to_datetime(df2023_periodo['Data'])
df2025_periodo['Data'] = pd.to_datetime(df2025_periodo['Data'])

# Criar coluna 'Semana' (ISO week)
df2023_periodo['Semana'] = df2023_periodo['Data'].dt.isocalendar().week
df2025_periodo['Semana'] = df2025_periodo['Data'].dt.isocalendar().week

# Média geral por ano (somando todos os distritos e semanas)
media_geral_2023 = df2023_periodo['Media_Classe'].mean()
media_geral_2025 = df2025_periodo['Media_Classe'].mean()

# Criar DataFrame para plot
media_geral = pd.DataFrame({
    'Ano': ['2023', '2025'],
    'Media_Classe': [media_geral_2023, media_geral_2025]
})

# Plot do gráfico geral

plt.figure(figsize=(6,5))
ax = sns.barplot(
    data=media_geral,
    x='Ano',
    y='Media_Classe',
    palette=['#FFA500', '#1F77B4'],  # Laranja=2023, Azul=2025
    alpha=0.8
)
# Adicionar valores em cima das barras
for p in ax.patches:
    ax.annotate(
        f'{p.get_height():.2f}',       # Formato com 2 casas decimais
        (p.get_x() + p.get_width() / 2., p.get_height()),  # posição central
        ha='center', va='bottom',
        fontsize=11
    )

plt.title('Média geral da classe por ano')
plt.xlabel('Ano')
plt.ylabel('Média da Classe')
plt.ylim(0, media_geral['Media_Classe'].max() * 1.15)  # Espaço no topo para as etiquetas
plt.tight_layout()
plt.show()

# %% [markdown]
# Percebemos que houve uma pequena redução da qualidade do ar comparando a média de 2023 e 2025.

# %% [markdown]
# #### Relação entre a Meteorologia e a Média da Qualidade do Ar

# %%
# Preparar dados de qualidade do ar
# Normalizar coluna de data (apenas dia)
df_ar["Data"] = pd.to_datetime(df_ar["Data"]).dt.floor("D")
df_ar["Dia"] = df_ar["Data"].dt.date  # coluna para merge por dia

# Preparar dados meteorológicos
# Converter para datetime (mantendo hora) e remover timezone
df_meteo["data"] = pd.to_datetime(df_meteo["date"], utc=True).dt.tz_convert(None)

df_meteo["Dia"] = df_meteo["data"].dt.date  # coluna para merge por dia

# Normalizar nomes dos distritos
df_meteo["distrito"] = df_meteo["distrito"].str.strip().str.title()

# Merge dos dados
df_combinadometeoqualar = pd.merge(
    df_ar,
    df_meteo,
    left_on=["Distrito", "Dia"],
    right_on=["distrito", "Dia"],
    how="left"
)

# Visualizar resultado
df_combinadometeoqualar


# %%
df_combinadometeoqualar.isna().any().any()

# %%
# Seleciona as colunas relevantes
variaveis_meteo = ["temperature_2m", "relative_humidity_2m", "rain", "wind_speed_80m", "Media_Classe"]

# Remove linhas com valores nulos
df_corr = df_combinadometeoqualar[variaveis_meteo].dropna()

# Calcula a matriz de correlação
matriz_corr = df_corr.corr()

# Cria o heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
plt.title("Correlação entre Meteorologia e Qualidade do Ar (2025)")
plt.show()

# %% [markdown]
# Este gráfico mostra que, em 2025, as variáveis meteorológicas têm impacto limitado sobre a média da qualidade do ar, com humidade sendo o fator mais relevante. Para uma análise mais profunda, seria interessante cruzar estes dados com tipos específicos de poluentes do ar.

# %% [markdown]
# ### Relação entre a Meteorologia e as Componentes da Qualidade do Ar

# %%
#Selecionar colunas relevantes
colunas_meteo = [
    "temperature_2m", "relative_humidity_2m", "rain",
    "temperature_80m", "wind_speed_80m", "wind_direction_80m",
    "temperature_2m_max", "temperature_2m_min", "uv_index_max"
]

#Verificar se todas as colunas estão presentes
colunas_disponiveis = [c for c in colunas_meteo + poluentesclean if c in df_combinadometeoqualar.columns]
if len(colunas_disponiveis) < len(colunas_meteo + poluentesclean):
    print("⚠️ Atenção: algumas colunas não foram encontradas no dataframe final.")
    print("Colunas encontradas:", colunas_disponiveis)

#Calcular correlação
corr = df_combinadometeoqualar[colunas_disponiveis].corr()

#Selecionar apenas as correlações Meteorologia × Poluentes
corr_sub = corr.loc[
    [c for c in colunas_meteo if c in corr.index],
    [c for c in poluentesclean if c in corr.columns]
]

#Criar heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr_sub,
    annot=True, fmt=".2f", cmap="coolwarm", center=0,
    linewidths=0.5, cbar_kws={'label': 'Correlação'}
)
plt.title("Correlação entre Variáveis Meteorológicas e Poluentes do Ar (2025)", fontsize=14, pad=15)
plt.xlabel("Poluentes do Ar")
plt.ylabel("Variáveis Meteorológicas")
plt.tight_layout()
plt.show()

# %% [markdown]
# Este heatmap mostra como variáveis meteorológicas influenciam os poluentes do ar em 2025. O ozono (O3) é o mais sensível: aumenta com temperaturas altas e radiação UV, mas diminui com maior humidade. As partículas (PM2.5 e PM10) reduzem com a chuva, que atua como “limpeza” atmosférica. O NO2 cresce em dias mais quentes, enquanto o SO2 não apresenta correlações fortes. O vento tem pouca influência direta. Em resumo, temperatura, radiação solar e chuva são os fatores mais relevantes para a qualidade do ar.

# %% [markdown]
# # Machine Learning

# %% [markdown]
# ### Juntar os dois dataset da Meteorologia com a Qualidade do Ar

# %%
# Obter lista de distritos presentes em df_mediaar
distritos_validos = df_ar['Distrito'].unique()

# Filtrar df_meteo para manter apenas esses distritos
df_meteo_filtrado = df_meteo[df_meteo['distrito'].isin(distritos_validos)]

# %%
print("Datas no df1:", df_meteo_filtrado["date"].max())

# %%
# Renomear colunas para uniformizar
df_ar = df_ar.rename(columns={'Data': 'date', 'Distrito': 'distrito'})
df_ar['date'] = pd.to_datetime(df_ar['date'], errors="coerce")
print(df_meteo_filtrado['date'].dtype)
print(df_meteo_filtrado['date'].head())
df_meteo_filtrado['date'] = pd.to_datetime(df_meteo_filtrado['date'], errors="coerce", utc=True)
# Converter para diário (ex: média por dia)
df_meteo_filtrado['date'] = df_meteo_filtrado['date'].dt.floor('D')
df_meteo_filtrado = df_meteo_filtrado.groupby('date').mean(numeric_only=True).reset_index()



# %%
print("df_ar date dtype:", df_ar["date"].dtype)
print("df_meteo_filtrado date dtype:", df_meteo_filtrado["date"].dtype)

# %%
df_meteo_filtrado["date"] = df_meteo_filtrado["date"].dt.tz_localize(None)

# %%
# Merge pelos campos comuns: 'date' e 'distrito'
df_merged = pd.merge(df_ar, df_meteo_filtrado, on=['date'], how='inner')

df_merged.columns

# %%
print(df_merged.isna().sum())

# %%
df_Model = df_merged.dropna()
print(df_Model.isna().sum())

# %%
df_Model

# %% [markdown]
# ### Vamos criar um modelo para o distrito de Lisboa

# %%
distritos = df_Model["distrito"].unique()

for d in distritos:
    nome_var = f"df_Model_{d}"
    globals()[nome_var] = df_Model[df_Model["distrito"] == d].copy()

# %%
df_Model_Lisboa

# %% [markdown]
# ## Vamos prever a qualidade do ar

# %%
# Filtrar Lisboa e copiar para evitar alterar o original
dfL = df_Model[df_Model["distrito"] == "Lisboa"].copy()

# Ordenar por Data (série temporal)
dfL["date"] = pd.to_datetime(dfL["date"], errors="coerce")
dfL = dfL.sort_values("date").reset_index(drop=True)

# Definir features e targets (excluindo CO e C6H6)
X_cols = [
    "rain", "temperature_2m", "relative_humidity_2m",
    "temperature_80m", "wind_speed_80m", "wind_direction_80m",
    "temperature_2m_max", "temperature_2m_min"
]
Y_cols = ["O3", "NO2", "SO2", "PM10", "PM2.5"]

X = dfL[X_cols]
# Vamos começar por um poluente de cada vez; por ex. PM10:
y = dfL["PM10"]

# Checar tipos e nulos (sanity check)
print(X.dtypes)
print(X.isna().sum())
print(y.isna().sum())

# %%
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

# %%
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    "LightGBM": LGBMRegressor(n_estimators=100, learning_rate=0.1, num_leaves=31, random_state=42),
    "MLP": MLPRegressor(hidden_layer_sizes=(64,32), alpha=0.001, max_iter=500, random_state=42)
}

# %%
results = []

for target in Y_cols:
    y = dfL[target].dropna()
    X = dfL[X_cols].loc[y.index]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    for name, model in models.items():
        grid = GridSearchCV(model, param_grids[name], cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Poluente": target,
            "Modelo": name,
            "BestParams": grid.best_params_,
            "MAE": mae,
            "R2": r2
        })

df_results = pd.DataFrame(results)
print(df_results)

# %% [markdown]
# **O3:**  MLP (MAE 15.66, R² 0.17) Justificativa: único com R² positivo e erro relativamente baixo.
# 
# **NO2:**  MLP (MAE 13.42, R² 0.30) Justificativa: melhor desempenho claro, explica ~30% da variância.
# 
# **SO2:**  LightGBM (MAE 0.645, R² 0.22) Justificativa: menor erro e R² positivo, consistente.
# 
# **PM10:**  RandomForest (MAE ~3.3, R² ~0.09–) Justificativa: R² ligeiramente positivo.
# 
# **PM2.5:**  MLP (MAE 6.47, R² -0.23) Justificativa: apesar do R² negativo, tem o menor erro absoluto.

# %% [markdown]
# Vamos adicionar:
# - Lags (valores desafados) que capturam dependência temporal de 1 e 2 passos atrás
# - Médias móveis, que suavizam variações e capturam tendência local (janela de 3)
# - Sazonalidade, que captura padrões mensais e semanais
# 
# Os lags são valores defasados da própria série que permitem capturar dependências temporais, como o valor de um ou dois dias anteriores, ajudando o modelo a relacionar o presente com o passado imediato. As médias móveis suavizam as variações diárias e destacam tendências locais, como no caso de uma janela de três dias que reduz o ruído e evidencia o movimento geral da série. A sazonalidade representa padrões que se repetem em ciclos regulares, como semanas ou meses, permitindo que os modelos reconheçam comportamentos recorrentes, por exemplo, maior poluição em determinados dias da semana ou em certas épocas do ano.
#  

# %%
#Adicionar lags temporais, para capturar dependência temporal:
for col in Y_cols:
    dfL[f"{col}_lag1"] = dfL[col].shift(1)
    dfL[f"{col}_lag2"] = dfL[col].shift(2)

#Criar médias móveis, para suavizar ruído e mostrar tendência:
for col in Y_cols:
    dfL[f"{col}_roll3"] = dfL[col].rolling(3).mean()

#Variáveis temporais, mês, dia da semana, estação:
dfL["month"] = dfL["date"].dt.month
dfL["weekday"] = dfL["date"].dt.weekday

# %%
X_cols = X_cols + \
    [f"{col}_lag1" for col in Y_cols] + \
    [f"{col}_lag2" for col in Y_cols] + \
    [f"{col}_roll3" for col in Y_cols] + \
    ["month", "weekday"]

# %%
#Normalização, antes de treinar SVR, MLP e ElasticNet:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%
#Validação temporalfrom sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)
grid = GridSearchCV(model, param_grids[name], cv=tscv, scoring="neg_mean_absolute_error", n_jobs=-1)


# %%
def plot_real_vs_pred(y_test, y_pred, poluente, modelo):
    """
    Gera gráfico comparativo entre valores reais e previstos.
    """
    plt.figure(figsize=(8,5))
    plt.plot(y_test.index, y_test.values, label="Real", marker="o")
    plt.plot(y_test.index, y_pred, label="Previsto", marker="x")
    plt.title(f"{poluente} — {modelo}: Real vs Previsto")
    plt.xlabel("Data")
    plt.ylabel(poluente)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# %%
results1 = []

for target in Y_cols:
    y = dfL[target].dropna()
    imputer = SimpleImputer(strategy="mean")  # ou "median"
    X = dfL[X_cols].loc[y.index]
    X = imputer.fit_transform(X)


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    for name, model in models.items():
        grid = GridSearchCV(model, param_grids[name], cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results1.append({
            "Poluente": target,
            "Modelo": name,
            "BestParams": grid.best_params_,
            "MAE": mae,
            "R2": r2
        })
    # Depois de calcular resultados para todos os modelos do poluente
    df_target = pd.DataFrame(results1).query("Poluente == @target")
    best_row = df_target.loc[df_target["R2"].idxmax()]
    best_model_name = best_row["Modelo"]
    best_params = best_row["BestParams"]

    # Re-treinar o melhor modelo com os melhores parâmetros
    model = models[best_model_name].set_params(**best_params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)



df_results1 = pd.DataFrame(results1)
print(df_results1)

# %% [markdown]
# **O3:** LightGBM (MAE 18.29, R² -0.44) Justificativa: apesar de todos os modelos apresentarem R² negativo, o LightGBM teve o menor erro absoluto, sendo o mais aceitável para este poluente.
# 
# **NO2:** LightGBM (MAE 14.31, R² 0.21) Justificativa: melhor desempenho claro, com menor erro e R² positivo, explicando cerca de 21% da variância.
# 
# **SO2:** LightGBM (MAE 0.43, R² 0.64) Justificativa: apresentou o menor erro e o maior R² positivo, mostrando excelente capacidade de previsão para este poluente.
# 
# **PM10:** LightGBM (MAE 3.11, R² 0.31) Justificativa: teve o menor erro e o maior R² positivo, conseguindo explicar parte relevante da variabilidade.
# 
# **PM2.5:** RandomForest (MAE 7.03, R² -0.49) Justificativa: embora todos os modelos tenham R² negativo, o RandomForest apresentou o menor erro absoluto, sendo o mais adequado neste caso.

# %% [markdown]
# #### Vamos tentar prever as classes dos componentes

# %%
df_class = dfL.copy()


# %%
fold_metrics = []

results_class = []

tscv = TimeSeriesSplit(n_splits=5)

targets_class = ["PM10_classe", "PM2.5_classe", "NO2_classe", "O3_classe", "SO2_classe"]

models = [
    (RandomForestClassifier(), "RandomForest"),
    (LogisticRegression(max_iter=500), "LogisticRegression"),
    (SVC(kernel="rbf"), "SVM"),
    (GaussianNB(), "NaiveBayes"),
    (KNeighborsClassifier(n_neighbors=5), "KNN"),
    (GradientBoostingClassifier(), "GradientBoosting"),
    (DecisionTreeClassifier(), "DecisionTree"),
    (ExtraTreesClassifier(), "ExtraTrees"),
    (MLPClassifier(max_iter=500), "MLP")
]

for target in targets_class:
    y = df_class[target].dropna()
    X = df_class[X_cols].loc[y.index].fillna(method="ffill").fillna(method="bfill")

    for clf, name in models:
        accs, f1s = [], []
        fold_idx = 0  # inicializa contador
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            if len(np.unique(y_train)) < 2:
                continue  # ignora fold sem diversidade de classes

            # Garantir que são classes discretas
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            accs.append(accuracy_score(y_test, y_pred))
            f1s.append(f1_score(y_test, y_pred, average="weighted"))

        results_class.append({
            "Classe": target,
            "Modelo": name,
            "Accuracy": np.mean(accs) if accs else None,
            "F1": np.mean(f1s) if f1s else None
        })

        fold_metrics.append({
            "Classe": target,
            "Modelo": name,
            "Fold": fold_idx,
            "Accuracy": accuracy_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred, average="weighted")
        })

        fold_idx += 1

df_results_class = pd.DataFrame(results_class)
print(df_results_class)
df_folds = pd.DataFrame(fold_metrics)
print(df_folds[df_folds["Classe"] == "Media_Classe"])


# %% [markdown]
# O motivo de não conseguirmos obter previsões para SO2_classe e PM2.5_classe podem ser os próprios dados, cujas variáveis de classe não têm diversidade suficiente, ou seja, praticamente todos os registros são iguais, vamos verificar:

# %%
for target in ["SO2_classe", "PM2.5_classe", "O3_classe"]:
    print(target, df_class[target].value_counts())

# %% [markdown]
# Como se confirma, as classes dos polunetes SO2 e PM2.5 são têm variabilidade suifiente, comparado por exemplo com O3.

# %% [markdown]
# #### Vamos tentar prever diretamente a Média das classes

# %% [markdown]
# Aqui normalizamos as variáveis para modelos sensíveis à escala e adicionamos uma baseline Naive Forecast, onde:
# 
# - O R2 é calculado em relação a prever a média.
# 
# - Se o Naive Forecast já é muito forte (porque Media_Classe varia pouco), os modelos precisam superar esse baseline.
# 
# - Isso torna mais evidente quando um modelo não acrescenta valor (R2 negativo).

# %%
target = "Media_Classe"
y = df_class[target].dropna().astype(float)
X = df_class[X_cols].loc[y.index].fillna(method="ffill").fillna(method="bfill")

# Modelos de regressão com normalização quando necessário
models = [
    (Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]), "LinearRegression"),
    (Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]), "Ridge"),
    (Pipeline([("scaler", StandardScaler()), ("model", Lasso(alpha=0.01))]), "Lasso"),
    (RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42), "RandomForestRegressor"),
    (GradientBoostingRegressor(n_estimators=200, max_depth=3, random_state=42), "GradientBoostingRegressor"),
    (ExtraTreesRegressor(n_estimators=200, max_depth=10, random_state=42), "ExtraTreesRegressor"),
    (Pipeline([("scaler", StandardScaler()), ("model", SVR(kernel="rbf", C=1.0, epsilon=0.1))]), "SVR"),
    (Pipeline([("scaler", StandardScaler()), ("model", MLPRegressor(hidden_layer_sizes=(50,50), max_iter=500, alpha=0.001, random_state=42))]), "MLPRegressor")
]

tscv = TimeSeriesSplit(n_splits=5)
results = []

# Baseline Naïve Forecast: prever o valor anterior
naive_maes, naive_rmses, naive_r2s = [], [], []
for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    y_pred_naive = y_train.shift(1).iloc[-len(y_test):].fillna(y_train.mean())
    naive_maes.append(mean_absolute_error(y_test, y_pred_naive))
    naive_rmses.append(np.sqrt(mean_squared_error(y_test, y_pred_naive)))
    naive_r2s.append(r2_score(y_test, y_pred_naive))

results.append({
    "Modelo": "NaiveForecast",
    "MAE": np.mean(naive_maes),
    "RMSE": np.mean(naive_rmses),
    "R2": np.mean(naive_r2s)
})

# Avaliar modelos
for clf, name in models:
    maes, rmses, r2s = [], [], []
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        maes.append(mean_absolute_error(y_test, y_pred))
        rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2s.append(r2_score(y_test, y_pred))

        # Bland-Altman plot
        mean_vals = (y_test + y_pred) / 2
        diff_vals = y_pred - y_test
        mean_diff = np.mean(diff_vals)
        std_diff = np.std(diff_vals)

        plt.scatter(mean_vals, diff_vals, alpha=0.7)
        plt.axhline(mean_diff, color='gray', linestyle='--', label='Mean difference')
        plt.axhline(mean_diff + 1.96 * std_diff, color='red', linestyle='--', label='+1.96 SD')
        plt.axhline(mean_diff - 1.96 * std_diff, color='red', linestyle='--', label='-1.96 SD')
        plt.xlabel("Mean of Real and Predicted")
        plt.ylabel("Difference (Predicted - Real)")
        plt.title(f"Bland-Altman: {name} - Fold {fold_idx}")
        plt.legend()
        plt.show()

    results.append({
        "Modelo": name,
        "MAE": np.mean(maes),
        "RMSE": np.mean(rmses),
        "R2": np.mean(r2s)
    })

df_results = pd.DataFrame(results).sort_values("R2", ascending=False)
print(df_results)


# %% [markdown]
# Para melhorar a capacidade preditiva e explorar dependências temporais na série, aplicámos a técnica de criação de lags diretamente na variável da média das classes. No caso, foram criados lags de 1 até 7 dias, de forma a capturar padrões semanais e possíveis persistências nos dados. As primeiras linhas, sem histórico suficiente, foram removidas para garantir consistência. Assim, o dataset fica enriquecido com atributos temporais que podem ajudar os modelos a identificar relações entre o passado e o presente da qualidade do ar.
# 
# Este método consiste em gerar variáveis que representam os valores passados da própria série, permitindo que o modelo utilize a informação histórica como preditores.

# %%

df_ar = df_class.copy()

# Criar lags (valores passados da própria série)
for lag in range(1, 8):  # até 7 dias atrás
    df_ar[f"lag{lag}"] = df_ar["Media_Classe"].shift(lag)

# Remover linhas com NaN (primeiros dias sem lags completos)
df_ar = df_ar.dropna()

# %%
X = df_ar[[f"lag{i}" for i in range(1, 8)]]
y = df_ar["Media_Classe"]


# %%
# Usar os melhores hiperparâmetros 
model_ar = SVR(C=10, epsilon=0.1, gamma=0.01)
model_ar.fit(X, y)

# Avaliar in-sample
y_pred_in = model_ar.predict(X)
print("MAE:", mean_absolute_error(y, y_pred_in))
print("RMSE:", np.sqrt(mean_squared_error(y, y_pred_in)))
print("R2:", r2_score(y, y_pred_in))


# %% [markdown]
# O modelo apresenta erros baixos, mas o poder explicativo é limitado. Isso sugere que ele consegue prever valores próximos, mas não explica totalmente os padrões da série.

# %%
import plotly.graph_objects as go
pio.renderers.default = "browser"
# Selecionar datas e valores reais
dates = df_class["date"].iloc[len(df_class)-len(y_pred_in):]
real_values = df_class["Media_Classe"].iloc[len(df_class)-len(y_pred_in):]

# Criar figura interativa
fig = go.Figure()

# Série real (azul)
fig.add_trace(go.Scatter(
    x=dates,
    y=real_values,
    mode="lines",
    name="Real",
    line=dict(color="blue")
))

# Série prevista (vermelho)
fig.add_trace(go.Scatter(
    x=dates,
    y=y_pred_in,
    mode="lines",
    name="Previsto (SVR autoregressivo)",
    line=dict(color="red")
))

# Layout
fig.update_layout(
    title="SVR autoregressivo - Real vs Previsto",
    xaxis_title="Data",
    yaxis_title="Media_Classe",
    hovermode="x unified",
    template="plotly_white",
    width=900,
    height=500
)

# Mostrar gráfico interativo
fig.show()
fig.write_html("grafico_svr.html")


# %%
import plotly.graph_objects as go

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_ar["date"].iloc[-len(y_pred_in):],
    y=df_ar["Media_Classe"].iloc[-len(y_pred_in):],
    mode='lines',
    name='Real',
    line=dict(color='navy')
))

fig.add_trace(go.Scatter(
    x=df_ar["date"].iloc[-len(y_pred_in):],
    y=y_pred_in,
    mode='lines',
    name='Previsto (SVR)',
    line=dict(color='red')
))

fig.update_layout(
    title="SVR Autoregressivo — Real vs Previsto",
    xaxis_title="Data",
    yaxis_title="Media_Classe",
    hovermode="x unified",
    template="plotly_white",
    height=500
)
fig.write_html("grafico_svr.html")
fig.show()



