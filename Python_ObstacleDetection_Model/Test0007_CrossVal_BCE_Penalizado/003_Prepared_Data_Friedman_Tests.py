import pandas as pd
import os

def preparar_dados_friedman(caminho_entrada, metrica):
    df = pd.read_csv(caminho_entrada)

    # Pesos para o cálculo da métrica composta (Weighted Score)
    pesos = {
        'Accuracy': 0.1,
        'Precision': 0.2,
        'Recall': 0.4,
        'F1-Score': 0.2,
        'ROC-AUC': 0.1
    }

    # Cálculo da métrica alvo
    if metrica == "Weighted":
        for key in pesos:
            if key not in df.columns:
                raise ValueError(f"[ERRO] A coluna '{key}' está faltando.")
            df[key] = pd.to_numeric(df[key], errors='coerce')
        df["Weighted Score"] = df[list(pesos.keys())].mul(pesos).sum(axis=1)
        metrica_col = "Weighted Score"
    else:
        if metrica not in df.columns:
            raise ValueError(f"[ERRO] A métrica '{metrica}' não foi encontrada.")
        df[metrica] = pd.to_numeric(df[metrica], errors='coerce')
        metrica_col = metrica

    # Define identificadores de configuração de modelo
    id_cols = [
        'ExtractModel', 'Pooling', 'Model_Parameters', 'Epochs', 'batch_size',
        'earlystop_patience', 'reduceLR_factor', 'reduceLR_patience',
        'loss_function', 'f1_alpha', 'f1_beta', 'peso_penalty_fn'
    ]

    df['Algoritmo'] = df[id_cols].astype(str).agg("_".join, axis=1)

    # Agrega por média para evitar duplicidade
    df_agrupado = df.groupby(['Algoritmo', 'Fold'], as_index=False)[metrica_col].mean()

    # Pivotagem
    pivot_df = df_agrupado.pivot(index='Algoritmo', columns='Fold', values=metrica_col)

    # Ordenação dos folds
    fold_cols = sorted(pivot_df.columns, key=lambda x: int(x))
    pivot_df = pivot_df[fold_cols]

    # Reanexa os metadados
    df_meta = df[['Algoritmo'] + id_cols].drop_duplicates().set_index('Algoritmo')
    pivot_df = df_meta.join(pivot_df)

    # Ordenação final
    final_cols = id_cols + fold_cols
    pivot_df = pivot_df[final_cols].reset_index(drop=True)

    # Salva resultado
    metrica_formatada = metrica.lower().replace(" ", "_")
    nome_arquivo_saida = f"friedman_prepared_data_{metrica_formatada}.csv"
    pivot_df.to_csv(nome_arquivo_saida, index=False)
    print(f"[INFO] Arquivo gerado com sucesso: {nome_arquivo_saida}")

# ================================
# Execução padrão
# ================================
if __name__ == "__main__":
    caminho_entrada = "BCE_Test_Results.csv"
    metrica = "F1-Score"  # Modifique conforme necessário: "Accuracy", "F1-Score", "Weighted", "Recall", "Precision" etc.
    preparar_dados_friedman(caminho_entrada, metrica)
