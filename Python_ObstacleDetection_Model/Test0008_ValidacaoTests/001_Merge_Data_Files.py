import pandas as pd

# Mapeamento dos campos usados para compor o campo 'model' para cada teste
model_fields = {
    "Test0004": [
        'ExtractModel', 'Pooling', 'Model_Parameters', 'Epochs',
        'Batch_Size', 'EarlyStop_Patience', 'ReduceLR_Factor', 'ReduceLR_Patience'
    ],
    "Test0005": [
        'ExtractModel', 'Pooling', 'Model_Parameters', 'Epochs',
        'Batch_Size', 'EarlyStop_Patience', 'ReduceLR_Factor',
        'reduceLR_patience', 'loss_function', 'f1_alpha', 'f1_beta', 'peso_penalty_fn'
    ],
    "Test0006": [
        'ExtractModel', 'Pooling', 'Model_Parameters', 'Epochs',
        'batch_size', 'earlystop_patience', 'reduceLR_factor',
        'reduceLR_patience', 'loss_function', 'f1_alpha', 'f1_beta', 'peso_penalty_fn'
    ],
    "Test0007": [
        'ExtractModel', 'Pooling', 'Model_Parameters', 'Epochs',
        'batch_size', 'earlystop_patience', 'reduceLR_factor',
        'reduceLR_patience', 'loss_function', 'f1_alpha', 'f1_beta', 'peso_penalty_fn'
    ]
}

# Caminhos dos arquivos CSV
file_paths = {
    "Test0004": "Test0004_MobileNetV1_CrossVal_Results_Merged.csv",
    "Test0005": "Test0005_MobileNetV1_CrossVal_Results_Merged.csv",
    "Test0006": "Test0006_F1_Loss_Test_Results.csv",
    "Test0007": "Test0007_BCE_Test_Results.csv"
}

# Lista para armazenar os DataFrames
dfs = []

# Processa cada arquivo e constrói a coluna 'model'
for test_id, path in file_paths.items():
    df = pd.read_csv(path)
    df["origem"] = test_id
    campos_modelo = model_fields[test_id]

    # Preenche campos ausentes com "None"
    for campo in campos_modelo:
        if campo not in df.columns:
            df[campo] = "None"

    # Gera o identificador único do modelo
    df["model"] = df[campos_modelo].astype(str).agg("_".join, axis=1)

    dfs.append(df)

# Junta todos os DataFrames
merged_df = pd.concat(dfs, ignore_index=True, sort=False)

# Seleciona e ordena colunas
colunas_base = ['origem', 'model', 'Fold']
colunas_metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
colunas_extra = [col for col in merged_df.columns if col not in colunas_base + colunas_metricas]

# Reorganiza o DataFrame
colunas_final = colunas_base + [col for col in colunas_metricas if col in merged_df.columns] + colunas_extra
merged_df = merged_df[colunas_final]

# Salva o resultado
merged_df.to_csv("Merged_Test_Results_Com_Model.csv", index=False)
print("[INFO] Arquivo salvo como: Merged_Test_Results_Com_Model.csv")
