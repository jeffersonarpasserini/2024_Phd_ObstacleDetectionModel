import pandas as pd
import os

def preparar_dados_friedman(caminho_entrada, metrica):
    df = pd.read_csv(caminho_entrada)

    # Define pesos para cálculo ponderado
    pesos = {
        'Accuracy': 0.15,
        'Precision': 0.15,
        'Recall': 0.30,
        'F1-Score': 0.25,
        'ROC-AUC': 0.15
    }

    # Seleciona a métrica
    if metrica == "Weighted":
        for key in pesos:
            if key not in df.columns:
                raise ValueError(f"[ERRO] A coluna '{key}' está faltando.")
            df[key] = pd.to_numeric(df[key], errors='coerce')
        df["Weighted Score"] = df[list(pesos.keys())].mul(pesos).sum(axis=1)
        metrica_col = "Weighted Score"
    else:
        if metrica not in df.columns:
            raise ValueError(f"[ERRO] A métrica '{metrica}' não foi encontrada no arquivo.")
        df[metrica] = pd.to_numeric(df[metrica], errors='coerce')
        metrica_col = metrica

    # Criação do identificador único para cada modelo
    id_cols = ['ExtractModel', 'Pooling', 'Model_Parameters', 'Epochs', 'Batch_Size',
               'EarlyStop_Patience', 'ReduceLR_Factor', 'ReduceLR_Patience']
    df['Algoritmo'] = df[id_cols].astype(str).agg("_".join, axis=1)

    # Pivotagem para deixar folds como colunas
    pivot_df = df.pivot(index='Algoritmo', columns='Fold', values=metrica_col)

    # Ordena os folds de 1 a 10, se existirem
    fold_cols = sorted(pivot_df.columns, key=lambda x: int(x))
    pivot_df = pivot_df[fold_cols]

    # Volta com os metadados originais
    df_meta = df[['Algoritmo'] + id_cols].drop_duplicates().set_index('Algoritmo')
    pivot_df = df_meta.join(pivot_df)

    # Reorganiza colunas
    final_cols = id_cols + fold_cols
    pivot_df = pivot_df[final_cols].reset_index(drop=True)

    # Gera nome do arquivo
    metrica_formatada = metrica.lower().replace(" ", "_")
    nome_arquivo_saida = f"friedman_prepared_data_{metrica_formatada}.csv"
    pivot_df.to_csv(nome_arquivo_saida, index=False)
    print(f"[INFO] Arquivo gerado com sucesso: {nome_arquivo_saida}")

# =========================
# Exemplo de uso
# =========================
if __name__ == "__main__":
    caminho_entrada = "MobileNetV1_CrossVal_Results_Merged.csv"
    metrica = "Accuracy"  # Ou "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Weighted"
    preparar_dados_friedman(caminho_entrada, metrica)
