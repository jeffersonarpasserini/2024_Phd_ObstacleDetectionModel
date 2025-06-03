import pandas as pd
import numpy as np
import ast

# Carregar os arquivos CSV
friedman_df = pd.read_csv('friedman_ranks.csv')
gridsearch_df = pd.read_csv('GridSearch_Tests_Results_MobilenetV1_avg_max.csv')

# Função para remover outliers e calcular a mediana
def median_wo_outliers(values):
    q1 = np.percentile(values, 25)
    q3 = np.percentile(values, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    filtered = [v for v in values if lower_bound <= v <= upper_bound]
    return np.median(filtered) if filtered else np.median(values)

# Processar cada linha do friedman_df
medians = []

for index, row in friedman_df.iterrows():
    try:
        params_str = row['params']
        # Encontrar todas as linhas do gridsearch com os mesmos parâmetros
        matching_rows = gridsearch_df[gridsearch_df['params'] == params_str]
        
        if not matching_rows.empty:
            acc_columns = [col for col in matching_rows.columns if 'split' in col and '_test_score' in col]
            scores = matching_rows[acc_columns].values.flatten()
            median = median_wo_outliers(scores)
        else:
            median = np.nan  # Se não encontrou o modelo

    except Exception as e:
        print(f"Erro ao processar linha {index}: {e}")
        median = np.nan

    medians.append(median)

# Adicionar a nova coluna no DataFrame original
friedman_df['median_accuracy_wo_outliers'] = medians

# Salvar em um novo CSV
friedman_df.to_csv('friedman_ranks_with_median.csv', index=False)

print("Arquivo salvo como 'friedman_ranks_with_median.csv'.")
