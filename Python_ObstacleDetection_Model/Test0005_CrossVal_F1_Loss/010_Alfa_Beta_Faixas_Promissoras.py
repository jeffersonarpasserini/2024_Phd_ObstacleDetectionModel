import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Função para carregar os arquivos de métricas
def carregar_metricas(caminhos):
    metricas = {}
    for nome, caminho in caminhos.items():
        df = pd.read_csv(caminho)
        metricas[nome] = df
    return metricas

# Função para gerar e salvar heatmaps e extrair top combinações
def gerar_heatmap_e_top_combinacoes(nome_metrica, df, top_n=5):
    df_f1 = df[df['loss_function'] == 'F1_Loss'].copy()
    df_f1['mean_score'] = df_f1.loc[:, '1':'10'].mean(axis=1)
    agrupado = df_f1.groupby(['f1_alpha', 'f1_beta'])['mean_score'].mean().reset_index()
    tabela_pivot = agrupado.pivot(index='f1_alpha', columns='f1_beta', values='mean_score')

    # Heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(tabela_pivot, annot=True, fmt=".4f", cmap='viridis')
    plt.title(f'Média da Métrica: {nome_metrica}')
    plt.xlabel('f1_beta')
    plt.ylabel('f1_alpha')
    plt.savefig(f"heatmap_{nome_metrica}.png")
    plt.close()

    # Top N combinações
    top_combinacoes = agrupado.sort_values('mean_score', ascending=False).head(top_n)
    return top_combinacoes

# Caminhos dos arquivos de métricas
caminhos_arquivos = {
    "accuracy": "friedman_prepared_data_accuracy.csv",
    "f1-score": "friedman_prepared_data_f1-score.csv",
    "precision": "friedman_prepared_data_precision.csv",
    "recall": "friedman_prepared_data_recall.csv",
    "weighted": "friedman_prepared_data_weighted.csv"
}

# Carregando os dados
metricas = carregar_metricas(caminhos_arquivos)

# Armazenar resultados promissores
melhores_faixas = {}

for nome_metrica, df in metricas.items():
    top_combinacoes = gerar_heatmap_e_top_combinacoes(nome_metrica, df)
    if nome_metrica in ["recall", "weighted", "f1-score"]:
        melhores_faixas[nome_metrica] = top_combinacoes

# Gerar arquivo de texto com as faixas mais promissoras
with open("faixas_promissoras.txt", "w") as f:
    for metrica, df in melhores_faixas.items():
        f.write(f"Métrica: {metrica.upper()}\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")

print("✅ Heatmaps salvos e faixas promissoras gravadas em faixas_promissoras.txt")
