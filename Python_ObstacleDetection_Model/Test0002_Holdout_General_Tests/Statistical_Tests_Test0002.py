import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare, normaltest
from scikit_posthocs import posthoc_nemenyi_friedman
from tqdm import tqdm, trange

class AlgorithmRanker:
    def __init__(self, data_path, enable_plots=True):
        """Inicializa com os dados do CSV."""
        print("[INFO] Carregando os dados...")
        self.original_data = pd.read_csv(data_path)
        self.data = None
        self.enable_plots = enable_plots
        self.weights = {
            'Accuracy': 0.15,
            'Precision': 0.15,
            'Recall': 0.30,
            'F1-Score': 0.25,
            'ROC-AUC': 0.15
        }

    def compute_weighted_score(self):
        """Calcula a pontuação ponderada de cada algoritmo."""
        print("[INFO] Calculando score ponderado...")
        self.data['Weighted Score'] = self.data[list(self.weights.keys())].mul(self.weights).sum(axis=1)

    def test_normality(self, metric):
        """Aplica o teste de normalidade global na métrica selecionada."""
        print(f"[INFO] Realizando teste de normalidade para a métrica: {metric}...")
        if metric not in self.data.columns:
            raise ValueError(f"Métrica {metric} não encontrada!")

        values = self.data[metric].dropna()
        stat, p = normaltest(values)

        if self.enable_plots:
            plt.figure(figsize=(10, 5))
            sns.kdeplot(values, fill=True)
            plt.title(f"Distribuição global da métrica {metric} (p={p:.8f})")
            plt.xlabel(metric)
            plt.ylabel("Densidade")
            plt.tight_layout()
            plt.savefig("normality_distribution_global.png")
            plt.close()

        with open("normality_test_results.txt", "w") as f:
            f.write(f"Global {metric}: stat = {stat:.8f}, p-value = {p:.8f}\n")

        return {'statistic': stat, 'p_value': p}

    def adjust_metric_based_on_distribution(self, metric):
        """Se os dados não forem normais, usa a mediana sem outliers como substituto da média."""
        print("[INFO] Ajustando valores com base na mediana sem outliers...")
        if 'Execucao' not in self.data.columns or 'Algoritmo' not in self.data.columns:
            raise ValueError("As colunas 'Execucao' e 'Algoritmo' são necessárias.")

        pivot = self.data.pivot(index='Execucao', columns='Algoritmo', values=metric)
        adjusted_scores = {}

        for col in tqdm(pivot.columns, desc="[INFO] Calculando medianas ajustadas"):
            q1 = pivot[col].quantile(0.25)
            q3 = pivot[col].quantile(0.75)
            iqr = q3 - q1
            filtered = pivot[col][(pivot[col] >= (q1 - 1.5 * iqr)) & (pivot[col] <= (q3 + 1.5 * iqr))]
            adjusted_scores[col] = filtered.median()

        for alg, val in adjusted_scores.items():
            self.data.loc[self.data['Algoritmo'] == alg, metric] = val

    def perform_friedman_test(self, metric):
        """Executa o Teste de Friedman e salva rankings e estatísticas."""
        print("[INFO] Executando teste de Friedman...")
        if metric not in self.data.columns:
            raise ValueError(f"Métrica {metric} não encontrada!")

        if 'Execucao' not in self.data.columns:
            raise ValueError("A coluna 'Execucao' é necessária para aplicar o teste de Friedman corretamente.")

        pivot = self.data.pivot(index='Execucao', columns='Algoritmo', values=metric)
        print(f"[DEBUG] Matriz de comparação (pivot) gerada com forma: {pivot.shape}")

        friedman_stat, p_value = friedmanchisquare(*[pivot[col].values for col in pivot.columns])

        avg_ranks = pivot.rank(axis=1, ascending=False).mean()
        avg_ranks.name = "Friedman Rank"

        self.data = self.data.merge(avg_ranks, left_on='Algoritmo', right_index=True, how='left')
        self.data.sort_values(by="Friedman Rank", inplace=True)

        nemenyi_results = None
        if p_value < 0.05:
            try:
                print("[INFO] Executando pós-teste de Nemenyi...")
                if pivot.shape[1] < 3:
                    print("[WARNING] Pós-teste de Nemenyi não pode ser aplicado com menos de 3 algoritmos.")
                else:
                    nemenyi_results = posthoc_nemenyi_friedman(pivot.values)
                    nemenyi_results.columns = pivot.columns
                    nemenyi_results.index = pivot.columns
                    nemenyi_results.to_csv("nemenyi_results.csv", index=True)
                    print("[INFO] Resultados do teste de Nemenyi salvos em 'nemenyi_results.csv'.")
            except Exception as e:
                print(f"[ERROR] Falha ao executar pós-teste de Nemenyi: {e}")
        else:
            print("[INFO] Pós-teste de Nemenyi não necessário (p >= 0.05).")

        with open("friedman_stats.txt", "w") as f:
            f.write(f"Friedman Test: X² = {friedman_stat:.8f}, p-value = {p_value:.8f}\n")

        avg_ranks.sort_values(inplace=True)
        avg_ranks.to_csv("friedman_ranks.csv", header=True)
        print("[INFO] Resultados do ranking salvos em 'friedman_ranks.csv'.")

        return friedman_stat, p_value

    def save_results(self, output_path):
        """Salva os resultados em um arquivo CSV ordenado por melhor rank."""
        self.data.sort_values(by="Friedman Rank", inplace=True)
        self.data.to_csv(output_path, index=False)
        print(f"[INFO] Resultados salvos em {output_path}")

    def read_gridsearch_results(self):
        """Lê e processa arquivo de GridSearch com múltiplos splits."""
        print("[INFO] Lendo resultados do GridSearch...")
        df = self.original_data.copy()
        score_columns = [col for col in df.columns if col.startswith("split") and col.endswith("_test_score")]

        melted = df.melt(
            id_vars=['params'],
            value_vars=score_columns,
            var_name='Execucao',
            value_name='Accuracy'
        )

        melted['Execucao'] = melted['Execucao'].str.extract(r'split(\d+)_test_score').astype(int)
        melted['Algoritmo'] = melted['params']

        melted.drop_duplicates(subset=['Execucao', 'Algoritmo'], inplace=True)

        melted['Precision'] = melted['Accuracy']
        melted['Recall'] = melted['Accuracy']
        melted['F1-Score'] = melted['Accuracy']
        melted['ROC-AUC'] = melted['Accuracy']

        self.data = melted
        print("[INFO] Dados prontos para análise.")

    def read_holdout_results(self):
        """Lê resultados provenientes de testes Holdout."""
        print("[INFO] Lendo resultados do Holdout...")
        df = self.original_data.copy()

        rename_map = {
            'Parâmetros': 'params',
            'Acurácia': 'Accuracy',
            'Precisão': 'Precision',
            'Recall': 'Recall',
            'F1-Score': 'F1-Score',
            'ROC-AUC': 'ROC-AUC'
        }
        df.rename(columns=rename_map, inplace=True)

        expected_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        for metric in expected_metrics:
            if metric not in df.columns:
                raise ValueError(f"Coluna esperada não encontrada: {metric}")

        df['Execucao'] = 0
        df['Algoritmo'] = df['params']

        self.data = df
        print("[INFO] Dados prontos para análise.")

# Exemplo de uso
if __name__ == "__main__":
    ENABLE_PLOTS = True  # Defina como False se não quiser salvar os gráficos
    USE_HOLDOUT = True   # Defina como True para usar resultados de holdout

    print("[INFO] Iniciando avaliação estatística de modelos...")
    file_path = "Holdout_Tests_Results_Summary_MobilenetV1_avg_max.csv" if USE_HOLDOUT else "GridSearch_Tests_Results_MobilenetV1_avg_max.csv"
    print("File: ", file_path)
    ranker = AlgorithmRanker(file_path, enable_plots=ENABLE_PLOTS)
    if USE_HOLDOUT:
        ranker.read_holdout_results()
    else:
        ranker.read_gridsearch_results()

    normality = ranker.test_normality("Accuracy")
    if normality['p_value'] < 0.05:
        print("[INFO] Distribuição não normal detectada. Ajustando métrica pela mediana sem outliers...")
        ranker.adjust_metric_based_on_distribution("Accuracy")
    else:
        print("[INFO] Distribuição normal detectada. Nenhum ajuste necessário.")

    ranker.compute_weighted_score()
    friedman_stat, p_value = ranker.perform_friedman_test("Accuracy")
    print(f"[INFO] Friedman Test: X² = {friedman_stat:.8f}, p-value = {p_value:.8f}")
    ranker.save_results("friedman_ranking.csv")
    print("[INFO] Processo finalizado com sucesso.")
