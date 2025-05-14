import pandas as pd

def calcular_mediana_sem_outliers(dados_folds):
    """
    Recebe uma Series com valores dos folds, remove outliers pelo método IQR e calcula a mediana.
    """
    valores = dados_folds.dropna()
    q1 = valores.quantile(0.25)
    q3 = valores.quantile(0.75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    filtrado = valores[(valores >= lim_inf) & (valores <= lim_sup)]
    return filtrado.median()

def processar_modelos(metrica="accuracy"):
    # Caminhos dos arquivos
    metrica = metrica.lower()
    arquivo_modelos = f"{metrica}_modelos_selecionados.csv"
    arquivo_dados = f"{metrica}_friedman_prepared_data_{metrica}.csv"
    arquivo_saida = f"{metrica}_modelos_selecionados_median.csv"

    # Lê os dados
    df_modelos = pd.read_csv(arquivo_modelos)
    df_dados = pd.read_csv(arquivo_dados)

    # Define colunas de identificação e folds conforme os novos arquivos
    id_cols = ['ExtractModel', 'Pooling', 'Parameters']
    fold_cols = [f"fold{i}" for i in range(1, 11)]

    # Cria coluna identificadora compatível com a dos modelos selecionados
    df_dados["Algoritmo"] = df_dados[id_cols].astype(str).agg("_".join, axis=1)

    # Filtra os dados para apenas os modelos selecionados
    df_filtrado = df_dados[df_dados["Algoritmo"].isin(df_modelos["Algoritmo"])].copy()

    # Converte os folds para numérico
    df_filtrado[fold_cols] = df_filtrado[fold_cols].apply(pd.to_numeric, errors='coerce')

    # Calcula a mediana sem outliers para cada linha
    df_filtrado["Median_Accuracy_wo_Outliers"] = df_filtrado[fold_cols].apply(calcular_mediana_sem_outliers, axis=1)

    # Prepara resultado final
    df_resultado = df_filtrado[id_cols + ["Median_Accuracy_wo_Outliers"]].copy()
    df_resultado["Algoritmo"] = df_resultado[id_cols].astype(str).agg("_".join, axis=1)

    # Junta com o Friedman Rank
    df_resultado = df_resultado.merge(df_modelos[["Algoritmo", "Friedman Rank"]], on="Algoritmo", how="left")

    # Reorganiza e ordena
    final_cols = id_cols + ["Friedman Rank", "Median_Accuracy_wo_Outliers"]
    df_resultado = df_resultado[final_cols].sort_values(by="Friedman Rank", ascending=True)

    # Salva arquivo
    df_resultado.to_csv(arquivo_saida, index=False)
    print(f"[INFO] Arquivo salvo em: {arquivo_saida}")

    return df_resultado

# Executa o processamento
if __name__ == "__main__":
    # Escolha da métrica analisada
    #metrica = "accuracy"
    # metrica = "precision"
    # metrica = "recall"
    # metrica = "f1-Score"
    # metrica = "ROC-AUC"
    metrica = "weighted"  # Descomente a métrica desejada

    processar_modelos(metrica)
