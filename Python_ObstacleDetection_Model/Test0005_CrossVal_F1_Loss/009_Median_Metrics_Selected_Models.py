import pandas as pd

def calcular_mediana_sem_outliers(series):
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    return series[(series >= lim_inf) & (series <= lim_sup)].median()

def calcular_metricas_por_modelo(df_dados, df_modelos, sufixo):

    pesos = {
        'Accuracy': 0.1,
        'Precision': 0.2,
        'Recall': 0.4,
        'F1-Score': 0.2,
        'ROC-AUC': 0.1
    }

    id_cols = ['ExtractModel', 'Pooling', 'Model_Parameters', 'Epochs', 'batch_size',
               'earlystop_patience', 'reduceLR_factor', 'reduceLR_patience',
               'loss_function', 'f1_alpha', 'f1_beta']
    metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    df_dados["Algoritmo"] = df_dados[id_cols].astype(str).agg("_".join, axis=1)
    df_modelos["Algoritmo"] = df_modelos["Algoritmo"].astype(str)

    for metrica in metricas:
        df_dados[metrica] = pd.to_numeric(df_dados[metrica], errors='coerce')

    resultado = []

    for _, linha in df_modelos.iterrows():
        modelo_id = linha["Algoritmo"]
        rank = linha["Friedman Rank"]
        subset = df_dados[df_dados["Algoritmo"] == modelo_id]

        entrada = {"Algoritmo": modelo_id, f"Friedman Rank_{sufixo}": rank}
        for metrica in metricas:
            entrada[f"{metrica}_Median_wo_Outliers_{sufixo}"] = calcular_mediana_sem_outliers(subset[metrica])

        entrada[f"Weighted_Median_wo_Outliers_{sufixo}"] = (
            entrada.get(f"Accuracy_Median_wo_Outliers_{sufixo}", 0) * 0.10 +
            entrada.get(f"Precision_Median_wo_Outliers_{sufixo}", 0) * 0.20 +
            entrada.get(f"Recall_Median_wo_Outliers_{sufixo}", 0) * 0.40 +
            entrada.get(f"F1-Score_Median_wo_Outliers_{sufixo}", 0) * 0.20 +
            entrada.get(f"ROC-AUC_Median_wo_Outliers_{sufixo}", 0) * 0.10
        )

        resultado.append(entrada)

    return pd.DataFrame(resultado)

# ==============================
# Execução principal
# ==============================
if __name__ == "__main__":
    # Arquivos de entrada
    arquivo_dados = "F1_Loss_Test_Results.csv"
    arquivo_accuracy = "accuracy_modelos_selecionados.csv"
    arquivo_weighted = "weighted_modelos_selecionados.csv"
    arquivo_saida = "comparativo_medianas_accuracy_weighted.csv"

    # Leitura
    df_dados = pd.read_csv(arquivo_dados)
    df_accuracy = pd.read_csv(arquivo_accuracy)
    df_weighted = pd.read_csv(arquivo_weighted)

    # Processamento separado
    df_acc = calcular_metricas_por_modelo(df_dados.copy(), df_accuracy, "Accuracy")
    df_wgt = calcular_metricas_por_modelo(df_dados.copy(), df_weighted, "Weighted")

    # Merge final por Algoritmo
    df_final = pd.merge(df_acc, df_wgt, on="Algoritmo", how="outer")
    df_final.to_csv(arquivo_saida, index=False)
    print(f"[INFO] Arquivo gerado com sucesso: {arquivo_saida}")
