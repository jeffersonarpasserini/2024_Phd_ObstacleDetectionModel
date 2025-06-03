import pandas as pd
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman

def executar_testes_estatisticos(caminho_entrada, metrica):
    print(f"[INFO] Lendo arquivo: {caminho_entrada}")
    df = pd.read_csv(caminho_entrada)

    # Garante que todas as colunas sejam strings
    df.columns = df.columns.astype(str)

    # Detecta colunas de fold numeradas de 1 a 10
    fold_cols = [str(i) for i in range(1, 11) if str(i) in df.columns]
    if len(fold_cols) < 2:
        raise ValueError("[ERRO] É necessário pelo menos 2 colunas de fold (1-10) para aplicar o teste de Friedman.")

    # Converte os folds para numérico
    for col in fold_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Novos identificadores do modelo conforme solicitado
    id_cols = [
        'ExtractModel', 'Pooling', 'Model_Parameters', 'Epochs', 'batch_size',
        'earlystop_patience', 'reduceLR_factor', 'reduceLR_patience',
        'loss_function', 'f1_alpha', 'f1_beta'
    ]
    id_cols_validas = [col for col in id_cols if col in df.columns]

    if not id_cols_validas:
        raise ValueError("[ERRO] Nenhuma coluna identificadora de modelo foi encontrada.")

    # Cria identificador do modelo
    df['Algoritmo'] = df[id_cols_validas].astype(str).agg("_".join, axis=1)

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
    metric_id = metrica.lower().replace("-", "_").replace(" ", "_")
    rank_file = f"{metric_id}_friedman_ranks.csv"
    stats_file = f"{metric_id}_friedman_stats.txt"
    nemenyi_file = f"{metric_id}_nemenyi_results.csv"

    # Salva os rankings
    avg_ranks_sorted.to_csv(rank_file, header=True)
    with open(stats_file, "w") as f:
        f.write(f"Friedman Test: X² = {friedman_stat:.8f}, p-value = {p_value:.8f}\n")

    # Pós-teste de Nemenyi
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
    # metrica = "Accuracy"
    # metrica = "Precision"
    # metrica = "Recall"
    # metrica = "F1-Score"
    # metrica = "ROC-AUC"
    metrica = "Weighted"  # Descomente a métrica desejada

    # Gera nome do arquivo de entrada
    metric_id = metrica.lower().replace("-", "_").replace(" ", "_")
    caminho_arquivo = f"friedman_prepared_data_{metric_id}.csv"

    # Executa os testes
    executar_testes_estatisticos(caminho_arquivo, metrica)
