import pandas as pd

def calcular_mediana_sem_outliers(dados_folds):
    """
    Remove outliers pelo método IQR e calcula a mediana.
    """
    valores = dados_folds.dropna()
    q1 = valores.quantile(0.25)
    q3 = valores.quantile(0.75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    filtrado = valores[(valores >= lim_inf) & (valores <= lim_sup)]
    return filtrado.median()

def processar_todas_metricas():
    metricas = ["accuracy", "precision", "recall", "f1-score", "roc-auc", "weighted"]
    fold_cols = [str(i) for i in range(1, 11)]
    id_cols = ['ExtractModel', 'Pooling', 'Model_Parameters', 'Epochs', 'Batch_Size',
               'EarlyStop_Patience', 'ReduceLR_Factor', 'ReduceLR_Patience']

    # Lê os modelos selecionados por accuracy e weighted
    df_acc = pd.read_csv("modelos_selecionados_accuracy.csv")
    df_weighted = pd.read_csv("modelos_selecionados_weighted.csv")

    # Junta os modelos (sem duplicar)
    df_modelos = pd.concat([df_acc, df_weighted]).drop_duplicates(subset=["Algoritmo"])

    # Lê o arquivo base onde as métricas serão adicionadas
    df_base = pd.read_csv("accuracy_vs_weighted_compared_friedman_tests.csv")

    for metrica in metricas:
        nome_metrica = metrica.lower().replace("-", "_")
        arquivo_dados = f"friedman_prepared_data_{metrica}.csv"

        try:
            df_dados = pd.read_csv(arquivo_dados)
        except FileNotFoundError:
            print(f"[AVISO] Arquivo {arquivo_dados} não encontrado. Pulando esta métrica.")
            continue

        # Cria a coluna identificadora dos modelos
        df_dados["Algoritmo"] = df_dados[id_cols].astype(str).agg("_".join, axis=1)

        # Filtra apenas os modelos selecionados
        df_filtrado = df_dados[df_dados["Algoritmo"].isin(df_modelos["Algoritmo"])].copy()
        df_filtrado[fold_cols] = df_filtrado[fold_cols].apply(pd.to_numeric, errors='coerce')

        # Calcula a mediana sem outliers
        df_filtrado.loc[:, nome_metrica + "_Median"] = df_filtrado[fold_cols].apply(calcular_mediana_sem_outliers, axis=1)

        # Adiciona essa métrica ao dataframe base
        df_base = pd.merge(df_base, df_filtrado[["Algoritmo", nome_metrica + "_Median"]],
                           on="Algoritmo", how="left")

    # Salva resultado
    df_base.to_csv("modelos_comparados_metricas_medianas.csv", index=False)
    print("[INFO] Arquivo final salvo como modelos_comparados_metricas_medianas.csv")

if __name__ == "__main__":
    processar_todas_metricas()
