import os
import pandas as pd


def merge_csvs_from_directory(directory_path, output_filename="merged_output.csv"):
    # Lista todos os arquivos CSV no diretório
    csv_files = sorted([f for f in os.listdir(directory_path) if f.endswith(".csv")])

    if not csv_files:
        print("Nenhum arquivo CSV encontrado no diretório.")
        return

    # Lista para armazenar os DataFrames
    dataframes = []

    for i, file_name in enumerate(csv_files):
        file_path = os.path.join(directory_path, file_name)

        if i == 0:
            # Lê o primeiro arquivo normalmente (com cabeçalho)
            df = pd.read_csv(file_path)
        else:
            # Pula a primeira linha (cabeçalho duplicado)
            df = pd.read_csv(file_path, skiprows=1, header=None)
            df.columns = dataframes[0].columns  # aplica o mesmo cabeçalho do primeiro

        dataframes.append(df)

    # Concatena todos os DataFrames
    merged_df = pd.concat(dataframes, ignore_index=True)

    # Caminho para salvar o arquivo final
    output_path = os.path.join(directory_path, output_filename)

    # Salva o resultado
    merged_df.to_csv(output_path, index=False)

    print(f"Arquivos mesclados com sucesso. Resultado salvo em: {output_path}")


if __name__ == "__main__":
    BASE_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))
    DATA_PATH = os.path.join(BASE_PATH, 'Test0006_CrossVal_F1_Loss_Penalizado', 'Results')
    print(DATA_PATH)
    merge_csvs_from_directory(DATA_PATH)
