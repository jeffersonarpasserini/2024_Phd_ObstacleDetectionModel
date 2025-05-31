import pandas as pd
import os

# Diretório onde estão os arquivos
caminho_arquivos = './'  # ou o caminho absoluto, como 'D:/Projetos/...'

# Nome base dos arquivos
base_nome = 'MobileNetV1_CrossVal_FullResults_'
extensao = '.csv'

# Lista para armazenar os DataFrames
lista_dfs = []

# Loop pelos arquivos de 1 a 22
for i in range(1, 21):
    nome_arquivo = f'{base_nome}{i:02d}{extensao}'
    caminho_completo = os.path.join(caminho_arquivos, nome_arquivo)

    # Se for o primeiro arquivo, lê com cabeçalho
    if i == 1:
        df = pd.read_csv(caminho_completo)
    else:
        df = pd.read_csv(caminho_completo, header=0)  # lê ignorando múltiplos cabeçalhos
    lista_dfs.append(df)

# Concatena todos os DataFrames
df_final = pd.concat(lista_dfs, ignore_index=True)

# Salva no novo arquivo
df_final.to_csv('MobileNetV1_CrossVal_Results_Merged.csv', index=False)
