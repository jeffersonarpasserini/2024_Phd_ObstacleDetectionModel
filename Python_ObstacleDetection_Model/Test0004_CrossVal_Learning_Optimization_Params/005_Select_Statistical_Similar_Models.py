import pandas as pd

def selecionar_modelos_equivalentes(metrica, top_n=20):
    # Prepara nomes de arquivos
    metrica_id = metrica.lower().replace("-", "_")
    ranks_csv = f"friedman_ranks_{metrica_id}.csv"
    nemenyi_csv = f"nemenyi_results_{metrica_id}.csv"
    output_csv = f"modelos_selecionados_{metrica_id}.csv"

    print(f"[INFO] Métrica selecionada: {metrica}")
    print(f"[INFO] Lendo arquivos: {ranks_csv} e {nemenyi_csv}")

    # Carrega os dados
    df_ranks = pd.read_csv(ranks_csv)
    nemenyi = pd.read_csv(nemenyi_csv, index_col=0)

    # Encontra o melhor modelo (menor rank)
    melhor_modelo = df_ranks.sort_values(by="Friedman Rank").iloc[0]["Algoritmo"]
    print(f"[INFO] Melhor modelo: {melhor_modelo}")

    # Modelos estatisticamente equivalentes (p >= 0.05)
    modelos_equivalentes = nemenyi[melhor_modelo][nemenyi[melhor_modelo] >= 0.05].index.tolist()
    total_equivalentes = len(modelos_equivalentes)

    print(f"[INFO] Encontrados {total_equivalentes} modelos estatisticamente equivalentes ao melhor.")

    # Filtra os top N entre os equivalentes
    df_filtrados = df_ranks[df_ranks["Algoritmo"].isin(modelos_equivalentes)].sort_values(by="Friedman Rank").head(top_n)

    # Salva resultado
    df_filtrados.to_csv(output_csv, index=False)
    print(f"[INFO] Top {top_n} modelos equivalentes salvos em: {output_csv}")
    print(f"[RESUMO] Foram encontrados {total_equivalentes} modelos equivalentes ao melhor, embora apenas os {top_n} melhores tenham sido salvos.")

    return df_filtrados


# ==============================
# Execução do script principal
# ==============================
if __name__ == "__main__":
    # Informe aqui a métrica desejada:
    # Exemplo: "Accuracy", "Recall", "F1-Score", "Weighted", etc.
    metrica = "Accuracy"
    top_n = 10  # Número de modelos a selecionar

    selecionar_modelos_equivalentes(metrica, top_n)
