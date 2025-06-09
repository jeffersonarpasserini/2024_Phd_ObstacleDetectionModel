import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def gerar_matriz_nemenyi(metrica="Accuracy", top_n=20):
    # Define os nomes dos arquivos com base na métrica
    metrica_id = metrica.lower().replace("-", "_")
    arquivo_nemenyi = f"{metrica_id}_nemenyi_results.csv"
    arquivo_selecionados = f"{metrica_id}_modelos_selecionados.csv"
    arquivo_saida = f"{metrica_id}_top{top_n}_nemenyi_matrix.csv"
    arquivo_saida_fig = f"{metrica_id}_top{top_n}_nemenyi_heatmap.png"

    print(f"[INFO] Lendo arquivos para métrica: {metrica}")
    df_nemenyi = pd.read_csv(arquivo_nemenyi, index_col=0)
    df_selecionados = pd.read_csv(arquivo_selecionados)

    # Extrai os modelos selecionados
    modelos = df_selecionados["Algoritmo"].tolist()

    # Verifica se todos os modelos existem na matriz de Nemenyi
    modelos_validos = [m for m in modelos if m in df_nemenyi.columns]
    if len(modelos_validos) < len(modelos):
        print("[WARNING] Alguns modelos selecionados não estão presentes na matriz de Nemenyi.")

    # Gera a submatriz
    matriz_topN = df_nemenyi.loc[modelos_validos, modelos_validos]

    # Mapear modelos para rótulos curtos Model_1, Model_2, ...
    rotulos_simplificados = {modelo: f"Model_{i+1}" for i, modelo in enumerate(modelos_validos)}
    matriz_topN_renomeada = matriz_topN.rename(index=rotulos_simplificados, columns=rotulos_simplificados)

    # Salva a matriz como CSV
    matriz_topN.to_csv(arquivo_saida)
    print(f"[INFO] Matriz de Nemenyi dos top {top_n} modelos salva em: {arquivo_saida}")

    # Gerar heatmap
    tamanho = max(10, len(modelos_validos) * 0.5)
    plt.figure(figsize=(tamanho, tamanho))

    sns.heatmap(matriz_topN_renomeada, annot=True, fmt=".3f", cmap="coolwarm", square=True,
                cbar_kws={'label': 'Distância Nemenyi'})

    plt.title(f"Matriz de Nemenyi — Top {top_n} modelos ({metrica})", fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(arquivo_saida_fig, dpi=300)
    plt.close()

    return matriz_topN

# =====================
# Execução do script
# =====================
if __name__ == "__main__":
    # Altere aqui conforme a métrica desejada
    # Escolha: "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", ou "Weighted"
    metrica = "Accuracy"
    top_n = 10 # deve corresponder ao arquivo de entrada

    gerar_matriz_nemenyi(metrica=metrica, top_n=top_n)
