import pandas as pd

def preparar_dados_friedman(caminho_entrada, metrica="F1-Score"):
    df = pd.read_csv(caminho_entrada)

    # Pesos para cálculo da métrica composta
    pesos = {
        'Accuracy': 0.1,
        'Precision': 0.2,
        'Recall': 0.4,
        'F1-Score': 0.2,
        'ROC-AUC': 0.1
    }

    # Calcular a métrica-alvo
    if metrica == "Weighted":
        for met in pesos:
            df[met] = pd.to_numeric(df[met], errors='coerce')
        df["Weighted Score"] = df[list(pesos.keys())].mul(pesos).sum(axis=1)
        metrica_col = "Weighted Score"
    else:
        if metrica not in df.columns:
            raise ValueError(f"[ERRO] Métrica '{metrica}' não encontrada.")
        df[metrica] = pd.to_numeric(df[metrica], errors='coerce')
        metrica_col = metrica

    # Usa o campo 'model' diretamente como identificador
    df["Algoritmo"] = df["model"]

    # Calcula média por modelo e fold
    df_agrupado = df.groupby(["Algoritmo", "Fold"], as_index=False)[metrica_col].mean()

    # Gera matriz modelo x fold
    pivot_df = df_agrupado.pivot(index="Algoritmo", columns="Fold", values=metrica_col)
    pivot_df = pivot_df[sorted(pivot_df.columns, key=lambda x: int(x))]

    # Adiciona metadados extras (origem)
    df_meta = df[["Algoritmo", "origem"]].drop_duplicates().set_index("Algoritmo")
    final_df = df_meta.join(pivot_df).reset_index()

    # Salva CSV de saída
    nome_saida = f"friedman_prepared_data_{metrica.lower().replace(' ', '_')}.csv"
    final_df.to_csv(nome_saida, index=False)
    print(f"[INFO] Arquivo salvo como: {nome_saida}")

# ================================
# Execução direta
# ================================
if __name__ == "__main__":
    # Altere a métrica conforme necessário: "Accuracy", "F1-Score", "Weighted", "Recall", "Precision", etc.
    metrica = "Recall"
    preparar_dados_friedman("Merged_Test_Results_Com_Model.csv", metrica)
