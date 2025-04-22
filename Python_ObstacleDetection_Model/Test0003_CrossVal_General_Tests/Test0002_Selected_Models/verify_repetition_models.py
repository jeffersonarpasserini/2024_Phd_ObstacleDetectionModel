import pandas as pd

file_path = "models_test001_0002.csv"
df = pd.read_csv(file_path)
print("Colunas disponíveis:", df.columns.tolist())

if 'Algoritmo' in df.columns:
    repetidos = df['Algoritmo'].value_counts()
    repetidos = repetidos[repetidos > 1]

    if not repetidos.empty:
        print("⚠️ Modelos repetidos encontrados:")
        print(repetidos)
    else:
        print("✅ Nenhum modelo repetido no campo 'Parâmetros'.")
else:
    print("❌ A coluna 'Parâmetros' não foi encontrada no arquivo.")


