import pandas as pd
from scipy.stats import friedmanchisquare
from scikit_posthocs import posthoc_nemenyi_friedman

def executar_testes_estatisticos(caminho_entrada, metrica):
    print(f"[INFO] Lendo arquivo: {caminho_entrada}")
    df = pd.read_csv(caminho_entrada)

    # Verifica e converte colunas de fold para string
    df.columns = df.columns.astype(str)

    # Detecta colunas de folds
    fold_cols = [col for col in df.columns if col.isdigit()]
    if len(fold_cols) < 2:
        raise ValueError("[ERRO] É necessário ao menos 2 folds para o teste de Friedman.")

    for col in fold_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Verifica existência de colunas obrigatórias
    if "Algoritmo" not in df.columns:
        raise ValueError("[ERRO] Coluna 'Algoritmo' não encontrada.")
    if "origem" not in df.columns:
        df["origem"] = "desconhecida"

    # Cria identificador único do modelo
    df["modelo_id"] = df["origem"] + " | " + df["Algoritmo"]

    # Monta matriz correta: modelo_id (linhas) × folds (colunas)
    matriz = df.set_index("modelo_id")[fold_cols]

    # Remove modelos com valores faltantes
    matriz_clean = matriz.dropna()
    print(f"[INFO] Dados utilizados: {matriz_clean.shape[0]} modelos × {matriz_clean.shape[1]} folds")

    # Teste de Friedman
    friedman_stat, p_value = friedmanchisquare(*[matriz_clean[col].values for col in matriz_clean.columns])
    print(f"[RESULTADO] Friedman Test: X² = {friedman_stat:.4f}, p-value = {p_value:.8f}")

    # Ranking médio
    avg_ranks = matriz_clean.rank(axis=0, ascending=False).mean(axis=1).sort_values()
    avg_ranks.name = "Friedman Rank"

    # Junta informações com origem
    df_resultado = avg_ranks.reset_index().rename(columns={"modelo_id": "model"})
    df_resultado["origem"] = df_resultado["model"].apply(lambda x: x.split(" | ")[0])

    # Arquivos de saída
    metric_id = metrica.lower().replace("-", "_").replace(" ", "_")
    stats_file = f"{metric_id}_friedman_stats.txt"
    rank_file = f"{metric_id}_friedman_ranks.csv"
    nemenyi_file = f"{metric_id}_nemenyi_results.csv"

    df_resultado.to_csv(rank_file, index=False)
    with open(stats_file, "w") as f:
        f.write(f"Friedman Test: X² = {friedman_stat:.8f}, p-value = {p_value:.8f}\n")

    # Pós-teste de Nemenyi
    # Pós-teste de Nemenyi com os 10 melhores modelos
    top_n = 10
    top_models = avg_ranks.head(top_n).index.tolist()
    matriz_top = matriz_clean.loc[top_models]

    print(f"[INFO] Executando pós-teste de Nemenyi nos {top_n} melhores modelos...")
    nemenyi = posthoc_nemenyi_friedman(matriz_top.values)
    nemenyi.columns = matriz_top.index
    nemenyi.index = matriz_top.index
    nemenyi.to_csv(nemenyi_file)
    print(f"[INFO] Pós-teste de Nemenyi salvo em: {nemenyi_file}")

    print("[INFO] Arquivos de saída:")
    print(f" - {stats_file}")
    print(f" - {rank_file}")
    print(f" - {nemenyi_file}")

    return friedman_stat, p_value

# Execução principal
if __name__ == "__main__":
    # Altere a métrica conforme necessário: "Accuracy", "F1-Score", "Weighted", "Recall", "Precision", etc.
    metrica = "Recall"
    metric_id = metrica.lower().replace("-", "_").replace(" ", "_")
    caminho_entrada = f"friedman_prepared_data_{metric_id}.csv"
    executar_testes_estatisticos(caminho_entrada, metrica)
