import pandas as pd
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman

def executar_testes_estatisticos(caminho_entrada, metrica):
    print(f"[INFO] Lendo arquivo: {caminho_entrada}")
    df = pd.read_csv(caminho_entrada)

    # Cria identificador de modelo
    df['Algoritmo'] = df['ExtractModel'] + "_" + df['Pooling'] + "_" + df['Parameters']

    # Detecta colunas de fold
    fold_cols = [col for col in df.columns if col.startswith("fold")]
    if len(fold_cols) < 2:
        raise ValueError("[ERRO] É necessário pelo menos 2 colunas de fold para aplicar o teste de Friedman.")

    # Monta matriz (linhas = execuções, colunas = algoritmos)
    matriz = df[['Algoritmo'] + fold_cols].copy()
    matriz.set_index('Algoritmo', inplace=True)
    pivot = matriz.T

    print(f"[INFO] Dados para análise: {pivot.shape[0]} execuções × {pivot.shape[1]} algoritmos")

    # Teste de Friedman
    friedman_stat, p_value = friedmanchisquare(*[pivot[col].values for col in pivot.columns])
    print(f"[RESULTADO] Friedman Test: X² = {friedman_stat:.4f}, p-value = {p_value:.8f}")

    # Ranking médio
    avg_ranks = pivot.rank(axis=1, ascending=False).mean()
    avg_ranks.name = "Friedman Rank"
    avg_ranks_sorted = avg_ranks.sort_values()

    # Nomes de saída personalizados
    metric_id = metrica.lower().replace("-", "_")
    rank_file = f"{metric_id}_friedman_ranks.csv"
    stats_file = f"{metric_id}_friedman_stats.txt"
    nemenyi_file = f"{metric_id}_nemenyi_results.csv"

    # Salva resultados
    avg_ranks_sorted.to_csv(rank_file, header=True)
    with open(stats_file, "w") as f:
        f.write(f"Friedman Test: X² = {friedman_stat:.8f}, p-value = {p_value:.8f}\n")

    if p_value < 0.05 and pivot.shape[1] >= 3:
        print("[INFO] Executando pós-teste de Nemenyi...")
        nemenyi = posthoc_nemenyi_friedman(pivot.values)
        nemenyi.columns = pivot.columns
        nemenyi.index = pivot.columns
        nemenyi.to_csv(nemenyi_file)
        print(f"[INFO] Pós-teste de Nemenyi salvo em: {nemenyi_file}")
    else:
        print("[INFO] Pós-teste de Nemenyi não necessário ou não aplicável.")

    print("[INFO] Arquivos salvos:")
    print(f" - {stats_file}")
    print(f" - {rank_file}")
    if p_value < 0.05:
        print(f" - {nemenyi_file}")

    return friedman_stat, p_value


# ========================
# Execução principal
# ========================
if __name__ == "__main__":
    # Escolha da métrica analisada
    metrica = "Weighted"  # selecione: "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Weighted"

    # Gera nome do arquivo de entrada
    metric_id = metrica.lower().replace("-", "_")
    caminho_arquivo = f"friedman_prepared_data_{metric_id}.csv"

    # Executa os testes
    executar_testes_estatisticos(caminho_arquivo, metrica)
