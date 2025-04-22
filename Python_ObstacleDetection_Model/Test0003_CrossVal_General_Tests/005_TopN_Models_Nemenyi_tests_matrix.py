import pandas as pd

def gerar_matriz_nemenyi(metrica="Accuracy", top_n=20):
    # Define os nomes dos arquivos com base na métrica
    metrica_id = metrica.lower().replace("-", "_")
    arquivo_nemenyi = f"{metrica_id}_nemenyi_results.csv"
    arquivo_selecionados = f"{metrica_id}_modelos_selecionados.csv"
    arquivo_saida = f"{metrica_id}_top{top_n}_nemenyi_matrix.csv"

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

    # Salva a matriz como CSV
    matriz_topN.to_csv(arquivo_saida)
    print(f"[INFO] Matriz de Nemenyi dos top {top_n} modelos salva em: {arquivo_saida}")

    return matriz_topN

# =====================
# Execução do script
# =====================
if __name__ == "__main__":
    # Altere aqui conforme a métrica desejada
    # Escolha: "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", ou "Weighted"
    metrica = "Weighted"
    top_n = 20  # deve corresponder ao arquivo de entrada

    gerar_matriz_nemenyi(metrica=metrica, top_n=top_n)
