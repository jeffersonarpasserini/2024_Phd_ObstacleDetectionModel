import pandas as pd
import numpy as np

def remover_outliers_e_medianar(df_metrica):
    """Remove outliers com base no IQR e retorna a mediana dos valores restantes."""
    q1 = df_metrica.quantile(0.25)
    q3 = df_metrica.quantile(0.75)
    iqr = q3 - q1
    filtro = ~((df_metrica < (q1 - 1.5 * iqr)) | (df_metrica > (q3 + 1.5 * iqr)))
    return df_metrica[filtro].median()

def gerar_metricas_para_ranking_composto():
    caminho_ranking = "ranking_composto_modelos.csv"
    caminho_dados = "Merged_Test_Results_Com_Model.csv"
    caminho_saida = "ranking_composto_modelos_com_metricas.csv"

    # Carrega os rankings compostos
    df_ranking = pd.read_csv(caminho_ranking)

    # Carrega os dados originais dos testes
    df_dados = pd.read_csv(caminho_dados, low_memory=False)

    # Lista de métricas a serem consideradas
    metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    # Pesos corretos fornecidos
    pesos = {
        'Accuracy': 0.1,
        'Precision': 0.2,
        'Recall': 0.4,
        'F1-Score': 0.2,
        'ROC-AUC': 0.1
    }

    # Gera campo 'modelo_id' compatível com o ranking
    df_dados['modelo_id'] = df_dados['origem'] + ' | ' + df_dados['model']

    # Remove outliers e calcula mediana para cada métrica
    medianas = df_dados.groupby("modelo_id")[metricas].apply(lambda g: remover_outliers_e_medianar(g)).reset_index()

    # Calcula 'Weighted' com os pesos definidos
    for metrica in metricas:
        if metrica not in medianas.columns:
            raise ValueError(f"[ERRO] Métrica ausente: {metrica}")

    medianas['Weighted'] = sum(medianas[m] * w for m, w in pesos.items())

    # Junta ao ranking
    df_ranking['modelo_id'] = df_ranking['model']
    df_final = pd.merge(df_ranking, medianas, on="modelo_id", how="left")

    # Salva o resultado
    df_final.to_csv(caminho_saida, index=False)
    print(f"[INFO] Arquivo salvo com sucesso: {caminho_saida}")

if __name__ == "__main__":
    gerar_metricas_para_ranking_composto()
import pandas as pd
import numpy as np

def remover_outliers_e_medianar(df_metrica):
    """Remove outliers com base no IQR e retorna a mediana dos valores restantes."""
    q1 = df_metrica.quantile(0.25)
    q3 = df_metrica.quantile(0.75)
    iqr = q3 - q1
    filtro = ~((df_metrica < (q1 - 1.5 * iqr)) | (df_metrica > (q3 + 1.5 * iqr)))
    return df_metrica[filtro].median()

def gerar_metricas_para_ranking_composto():
    caminho_ranking = "ranking_composto_modelos.csv"
    caminho_dados = "Merged_Test_Results_Com_Model.csv"
    caminho_saida = "ranking_composto_modelos_com_metricas.csv"

    # Carrega os rankings compostos
    df_ranking = pd.read_csv(caminho_ranking)

    # Carrega os dados originais dos testes
    df_dados = pd.read_csv(caminho_dados, low_memory=False)

    # Lista de métricas a serem consideradas
    metricas = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    # Pesos corretos fornecidos
    pesos = {
        'Accuracy': 0.1,
        'Precision': 0.2,
        'Recall': 0.4,
        'F1-Score': 0.2,
        'ROC-AUC': 0.1
    }

    # Gera campo 'modelo_id' compatível com o ranking
    df_dados['modelo_id'] = df_dados['origem'] + ' | ' + df_dados['model']

    # Remove outliers e calcula mediana para cada métrica
    medianas = df_dados.groupby("modelo_id")[metricas].apply(lambda g: remover_outliers_e_medianar(g)).reset_index()

    # Calcula 'Weighted' com os pesos definidos
    for metrica in metricas:
        if metrica not in medianas.columns:
            raise ValueError(f"[ERRO] Métrica ausente: {metrica}")

    medianas['Weighted'] = sum(medianas[m] * w for m, w in pesos.items())

    # Junta ao ranking
    df_ranking['modelo_id'] = df_ranking['model']
    df_final = pd.merge(df_ranking, medianas, on="modelo_id", how="left")

    # Salva o resultado
    df_final.to_csv(caminho_saida, index=False)
    print(f"[INFO] Arquivo salvo com sucesso: {caminho_saida}")

if __name__ == "__main__":
    gerar_metricas_para_ranking_composto()
