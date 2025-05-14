import pandas as pd

def comparar_rankings(arquivo_accuracy, arquivo_weighted, arquivo_saida):
    # Carrega os arquivos
    df_accuracy = pd.read_csv(arquivo_accuracy).rename(columns={"Friedman Rank": "Friedman Rank Accuracy"})
    df_weighted = pd.read_csv(arquivo_weighted).rename(columns={"Friedman Rank": "Friedman Rank Weighted"})

    # Merge pela coluna 'Algoritmo'
    df_comparado = pd.merge(df_accuracy, df_weighted, on="Algoritmo", how="outer")

    # Salva arquivo de saída
    df_comparado.to_csv(arquivo_saida, index=False)
    print(f"[INFO] Comparação salva em: {arquivo_saida}")

    return df_comparado

# =========================
# Execução principal
# =========================
if __name__ == "__main__":
    arquivo_accuracy = "accuracy_modelos_selecionados.csv"
    arquivo_weighted = "weighted_modelos_selecionados.csv"
    arquivo_saida = "accuracy_vs_weighted_compared_friedman_tests.csv"

    comparar_rankings(arquivo_accuracy, arquivo_weighted, arquivo_saida)
