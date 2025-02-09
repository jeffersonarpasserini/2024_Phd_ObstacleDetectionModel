import os

import pandas as pd


def process_csv(file_path, output_file):
    # Carregar o arquivo CSV
    df = pd.read_csv(file_path)

    # Selecionar as colunas dos splits para análise
    split_columns = [f'split{i}_test_score' for i in range(10)]

    # Calcular os quartis, limites de outliers e mediana sem outliers por linha
    df['Q1'] = df[split_columns].quantile(0.25, axis=1)
    df['Q2'] = df[split_columns].median(axis=1)
    df['Q3'] = df[split_columns].quantile(0.75, axis=1)

    df['IQR'] = df['Q3'] - df['Q1']
    df['Lower Bound'] = df['Q1'] - 1.5 * df['IQR']
    df['Upper Bound'] = df['Q3'] + 1.5 * df['IQR']

    # Filtrar os valores dentro dos limites para cada linha e calcular a nova mediana
    def filter_outliers(row):
        filtered_values = row[split_columns][
            (row[split_columns] >= row['Lower Bound']) & (row[split_columns] <= row['Upper Bound'])]
        return filtered_values.median() if not filtered_values.empty else row['Q2']

    df['Median Without Outliers'] = df.apply(filter_outliers, axis=1)

    # Criar a coluna de ranqueamento das medianas sem outliers (ordem decrescente)
    df['Ranking'] = df['Median Without Outliers'].rank(method='min', ascending=False)

    # Identificar a melhor amostra com base na mediana sem outliers
    best_sample = df.loc[df['Ranking'].idxmin(), ['params', 'Median Without Outliers']]

    # Remover colunas auxiliares de cálculo
    df.drop(columns=['IQR'], inplace=True)

    # Salvar o novo arquivo CSV
    df.to_csv(output_file, index=False)

    print(f"Arquivo processado e salvo em: {output_file}")
    print(f"Melhor amostra: {best_sample['params']}, Mediana sem outliers: {best_sample['Median Without Outliers']}")


# ----------------------- MAIN ------------------------------------------------
def main_calc_results():
    # Diretório base onde estão os arquivos CSV
    BASE_PATH = os.path.dirname(os.path.abspath(__file__))
    RESULT_PATH = os.path.join(BASE_PATH, 'C:\\Projetos\\2024_Phd_ObstacleDetectionModel\\Python_ObstacleDetection_Model\\Results_Test_GridSearch')

    # Percorrer todos os arquivos CSV no diretório e processá-los
    for file_name in os.listdir(RESULT_PATH):
        if file_name.endswith(".csv"):
            input_file_path = os.path.join(RESULT_PATH, file_name)
            output_file_path = os.path.join(RESULT_PATH, f"CALC_{file_name}")
            process_csv(input_file_path, output_file_path)


# Este bloco garante que o código seja executado apenas quando o arquivo for executado diretamente
if __name__ == "__main__":
    main_calc_results()