import pandas as pd

# Caminhos dos arquivos de entrada
arquivo_accuracy = "accuracy_friedman_ranks.csv"
arquivo_recall = "recall_friedman_ranks.csv"
arquivo_weighted = "weighted_friedman_ranks.csv"
#arquivo_f1_score = "f1_score_friedman_ranks.csv"

# Leitura dos arquivos com renomeação dos rankings
df_acc = pd.read_csv(arquivo_accuracy).rename(columns={"Friedman Rank": "Rank_Accuracy"})
df_rec = pd.read_csv(arquivo_recall).rename(columns={"Friedman Rank": "Rank_Recall"})
df_wgt = pd.read_csv(arquivo_weighted).rename(columns={"Friedman Rank": "Rank_Weighted"})
#df_f1  = pd.read_csv(arquivo_f1_score).rename(columns={"Friedman Rank": "Rank_F1_Score"})

# Verifica se as colunas essenciais existem
for df in [df_acc, df_rec, df_wgt]:
    if "model" not in df.columns or "origem" not in df.columns:
        raise ValueError("Os arquivos devem conter as colunas 'model' e 'origem'.")

# Mesclagem dos rankings por modelo + origem
df_merged = df_acc.merge(df_rec, on=["model", "origem"]) \
                  .merge(df_wgt, on=["model", "origem"])
#                  .merge(df_f1,  on=["model", "origem"])

# Cálculo do ranking composto (média dos três rankings)
df_merged["Rank_Composto"] = df_merged[[
    "Rank_Accuracy", "Rank_Recall", "Rank_Weighted"
]].mean(axis=1)

# Ordena pelo ranking composto
df_ordenado = df_merged.sort_values(by="Rank_Composto")

# Salva o resultado
df_ordenado.to_csv("ranking_composto_modelos.csv", index=False)

print("[INFO] Ranking composto com 3 métricas gerado e salvo como 'ranking_composto_modelos.csv'.")

