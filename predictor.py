import requests
import pandas as pd
from bs4 import BeautifulSoup
import re
import glob
import os
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pickle
from datetime import datetime
import tensorflow as tf



def get_answer(string):
  value = np.zeros(shape=(9))
  if string == 'Accumulate':
    value[0] = 1
  elif string == 'BUY!':
    value[1] = 1
  elif string == 'FOMO intensifies':
    value[2] = 1
  elif string == 'Fire sale!':
    value[3] = 1
  elif string == 'HODL':
    value[4] = 1
  elif string == 'Is it a bubble?':
    value[5] = 1
  elif string == 'Maximum bubble territory':
    value[6] = 1
  elif string == 'Sell. Seriously, sell!':
    value[7] = 1
  elif string == 'Still cheap':
    value[8] = 1
  else:
    pass
  return value

def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    return local_filename

def listar_arquivos(caminho_da_pasta):
    arquivos = glob.glob(os.path.join(caminho_da_pasta, '**'), recursive=True)
    return [arquivo for arquivo in arquivos if os.path.isfile(arquivo)]

# Função df_to_row para agregar os dados e retornar um novo DataFrame
def df_to_row(df):
    new_data = {
        'timestamp': pd.to_datetime(df['time'].iloc[0]).strftime('%Y-%m-%d'),
        'Weight Mean': df['weight'].mean(),
        'Difficulty Mean': df['difficulty'].mean(),
        'Reward Mean': df['reward_usd'].mean(),
        'Transaction Sum': df['transaction_count'].sum(),
        'Witness Sum': df['witness_count'].sum(),
        'Input Sum': df['input_count'].sum(),
        'Output Sum': df['output_count'].sum(),
        'Fee Total Sum': df['fee_total_usd'].sum(),
        'Total Blocks': len(df)
    }
    return pd.DataFrame([new_data])

def on_chain_df():
    # Exemplo de uso:
    caminho = './downloads/'
    arquivos = listar_arquivos(caminho)

    # Criação de um DataFrame vazio para armazenar os resultados
    df_result = pd.DataFrame()

    # Loop através dos arquivos para agregar os dados
    for i in range(len(arquivos)):
        try:
            df_nn = pd.read_csv(arquivos[i], sep='\t')
            row_to_append = df_to_row(df_nn)
            df_result = pd.concat([df_result, row_to_append], ignore_index=True)
        except Exception as e:
            print(f"Erro ao processar o arquivo {arquivos[i]}: {e}")
            pass

    return df_result

def get_the_prediction(lista_5_days):

  # Obtenha os dados intradiários
  url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
  parameters = {
      'vs_currency': 'usd',
      'days': '21',  # Número de dias para obter os dados
      'interval': 'daily'  # Obter dados diários
  }

  response = requests.get(url, params=parameters)
  data = response.json()

  # Convertendo para DataFrame
  df = pd.DataFrame({
      'timestamp': [pd.to_datetime(x[0], unit='ms').date() for x in data['prices']],
      'price': [x[1] for x in data['prices']],
      'market_cap': [x[1] for x in data['market_caps']],
      'volume': [x[1] for x in data['total_volumes']]
  })

  # Remover dados duplicados por data (se houver múltiplas entradas por dia)
  df = df.groupby('timestamp').agg({
      'price': 'last',  # Usar o último preço do dia como fechamento
      'market_cap': 'last',  # Usar o último market cap do dia como fechamento
      'volume': 'last'  # Usar o último volume do dia como fechamento
  }).reset_index()

  #onchain_data
  ano_atual = datetime.now().year

  # Fazendo uma requisição GET para a página
  url = 'https://gz.blockchair.com/bitcoin/blocks/'
  response = requests.get(url)

  # Criando um objeto BeautifulSoup para analisar o conteúdo HTML
  soup = BeautifulSoup(response.text, 'html.parser')

  html_base = 'https://gz.blockchair.com/bitcoin/blocks/'
  links = soup.find_all('a')

  padrao = r'>(.*?)<'
  lista_links = []
  for i in range(len(links[4:])):
      correspondencia = re.findall(padrao, str(links[4+i]))
      if str(ano_atual) in correspondencia[0]:
        lista_links.append(html_base + correspondencia[0])
      else:
        pass

  lista_links = lista_links[-22:]  

  # Diretório de destino para salvar os arquivos
  output_directory = 'downloads'

  # Criando o diretório de destino se não existir
  os.makedirs(output_directory, exist_ok=True)

  for link in lista_links:
    try:
      filename = os.path.join(output_directory, link.split('/')[-1])  # Extrai o nome do arquivo do URL
      download_file(link, filename)
    except:
      pass


  # Criação de um DataFrame vazio para armazenar os resultados
  df_result = on_chain_df()

  df_result['timestamp'] = pd.to_datetime(df_result['timestamp'])
  df['timestamp'] = pd.to_datetime(df['timestamp'])

  df_result = df_result.sort_values(by='timestamp')
  result_df = pd.merge(df, df_result, on='timestamp', how='inner')

  scaler = MinMaxScaler()
  x_scaled = scaler.fit_transform(result_df.drop(['timestamp'], axis=1))

  with open('./model_voting_soft_ml.pkl', 'rb') as file:
      modelo = pickle.load(file)

  preds = modelo.predict(x_scaled)

  count = 0
  class_of_rainbow = []
  for day in lista_5_days:
    x = get_answer(day)
    class_of_rainbow.append(x)
    count += 1
  
  # Obter os últimos 5 registros
  ultimos_5_X = x_scaled[-5:]
  ultimos_5_preds = preds[-5:]

  array_for_prediction = []

  for i in range(len(ultimos_5_X)):
    list_to_play = []
    for j in range(len(ultimos_5_X[i])):
      list_to_play.append(ultimos_5_X[i][j])
    list_to_play.append(ultimos_5_preds[i])
    for k in range(len(class_of_rainbow[i])):
      list_to_play.append(class_of_rainbow[i][k])
    array_for_prediction.append(list_to_play)

  #convert list to array:
  array_pred_normalized = np.array(array_for_prediction)
  array_expanded = np.expand_dims(array_pred_normalized, axis=0)

  # Carrega o modelo
  lstm_model = tf.keras.models.load_model('model_lstm_dl.h5')

  prediction_ltsm_stacked = np.argmax(lstm_model.predict(array_expanded), axis=1)

  oracle_indication = ''

  if prediction_ltsm_stacked[0] == 0:
    oracle_indication = 'Vai cair! (SELL)'
  else:
    oracle_indication = 'Vai subir! (BUY)'

  return oracle_indication