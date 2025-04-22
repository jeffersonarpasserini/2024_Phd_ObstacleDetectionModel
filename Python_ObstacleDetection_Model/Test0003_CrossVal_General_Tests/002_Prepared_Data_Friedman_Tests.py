import pandas as pd
import os

def preparar_dados_friedman(caminho_entrada, metrica):
    # Carrega o CSV original
    df = pd.read_csv(caminho_entrada)

    # Renomeia colunas, se necessário
    rename_map = {
        'Acurácia': 'Accuracy',
        'Precisão': 'Precision',
        'Parâmetros': 'Parameters',
        'Parâmetros_str': 'Parameters_str'
    }
    df.rename(columns=rename_map, inplace=True)

    # Define pesos para cálculo ponderado
    pesos = {
        'Accuracy': 0.15,
        'Precision': 0.15,
        'Recall': 0.30,
        'F1-Score': 0.25,
        'ROC-AUC': 0.15
    }

    # Escolhe a métrica
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

    # Extrai fold do campo Model (ex: fold1, fold2, ..., fold10)
    df['Fold'] = df['Model'].str.extract(r'_(fold\d+)_')
    df = df.dropna(subset=['Fold'])

    # Cria identificador único para o modelo
    df['Algoritmo'] = df['ExtractModel'].astype(str) + "_" + df['Pooling'].astype(str) + "_" + df['Parameters'].astype(str)

    # Pivot para transformar em fold1 ~ fold10
    pivot_df = df.pivot(index='Algoritmo', columns='Fold', values=metrica_col)

    # Garante colunas fold1 a fold10
    for i in range(1, 11):
        col = f'fold{i}'
        if col not in pivot_df.columns:
            pivot_df[col] = pd.NA

    # Ordena as colunas dos folds
    pivot_df = pivot_df[[f'fold{i}' for i in range(1, 11)]]

    # Separa os campos do identificador
    pivot_df.reset_index(inplace=True)
    pivot_df[['ExtractModel', 'Pooling', 'Parameters']] = pivot_df['Algoritmo'].str.split("_", n=2, expand=True)

    # Organiza colunas finais
    cols = ['ExtractModel', 'Pooling', 'Parameters'] + [f'fold{i}' for i in range(1, 11)]
    pivot_df = pivot_df[cols]

    # Cria nome do arquivo de saída com base na métrica
    metrica_formatada = metrica.lower().replace(" ", "_")
    nome_arquivo_saida = f"friedman_prepared_data_{metrica_formatada}.csv"

    # Salva o arquivo
    pivot_df.to_csv(nome_arquivo_saida, index=False)
    print(f"[INFO] Arquivo gerado com sucesso: {nome_arquivo_saida}")


# =========================
# Exemplo de uso
# =========================
if __name__ == "__main__":
    caminho_entrada = "MobileNetV1_crossval_results_detail.csv"

    # Escolha: "Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", ou "Weighted"
    metrica = "Weighted"

    preparar_dados_friedman(caminho_entrada, metrica)
