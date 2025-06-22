import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import friedmanchisquare

# ==========================
# CONFIGURAÇÕES
# ==========================

# Arquivos CSV
arquivo_dados = 'Merged_Test_Results_Com_Model.csv'
arquivo_ranking = 'ranking_composto_modelos.csv'

# 🔧 Escolha a métrica: 'Recall', 'Accuracy' ou 'Weighted'
metrica = 'Accuracy'

# Pasta de saída (opcional, pode ser '.')
pasta_saida = '.'

# ==========================
# LEITURA DO RANKING
# ==========================

# Carregar o ranking
df_ranking = pd.read_csv(arquivo_ranking)

# Selecionar os 5 melhores modelos
top5_modelos = df_ranking.sort_values(by='Rank_Composto').head(5)['model'].tolist()

print("\n🟩 5 Melhores Modelos Selecionados:")
for i, modelo in enumerate(top5_modelos, start=1):
    print(f"Model{i:02d}: {modelo}")

# Criar mapeamento de nome original para Model01, Model02...
modelo_labels = {modelo: f"Model{i:02d}" for i, modelo in enumerate(top5_modelos, start=1)}

# ==========================
# LEITURA DOS DADOS
# ==========================

# Carregar os dados brutos
df = pd.read_csv(arquivo_dados, low_memory=False)

# Criar identificador de modelo
df['modelo_id'] = df['origem'] + ' | ' + df['model']

# ==========================
# CÁLCULO DE WEIGHTED (se necessário)
# ==========================

if metrica == 'Weighted':
    pesos = {
        'Accuracy': 0.1,
        'Precision': 0.2,
        'Recall': 0.4,
        'F1-Score': 0.2,
        'ROC-AUC': 0.1
    }

    missing_columns = [m for m in pesos if m not in df.columns]
    if missing_columns:
        raise ValueError(f"❌ As seguintes métricas estão ausentes nos dados: {missing_columns}")

    df['Weighted'] = sum(df[m] * w for m, w in pesos.items())

# ==========================
# FILTRAGEM DOS MODELOS
# ==========================

# Filtrar apenas os modelos de interesse
df_filtrado = df[df['modelo_id'].isin(top5_modelos)]

# ==========================
# PREPARAÇÃO DA MATRIZ
# ==========================

# Pivotar: linhas = Fold, colunas = modelos, valores = métrica escolhida
pivot = df_filtrado.pivot_table(index='Fold', columns='modelo_id', values=metrica)

# Remover folds com dados ausentes (NaN)
pivot = pivot.dropna()

# Renomear as colunas usando Model01, Model02, ...
pivot.rename(columns=modelo_labels, inplace=True)

ordem_modelos = [modelo_labels[m] for m in top5_modelos]
pivot = pivot[ordem_modelos]

print("\n📊 Tabela de dados por fold:\n", pivot)

# ==========================
# TESTE DE FRIEDMAN
# ==========================

# Aplicar o teste de Friedman
stat, p_value = friedmanchisquare(*[pivot[col] for col in pivot.columns])

print("\n✅ Resultado do Teste de Friedman:")
print(f"Estatística: {stat:.4f}")
print(f"p-valor: {p_value:.4f}")

if p_value < 0.05:
    print("➡️ Existem diferenças estatisticamente significativas entre os modelos.")
else:
    print("➡️ NÃO existem diferenças estatisticamente significativas entre os modelos.")

# ==========================
# MATRIZ DE NEMENYI (DIFERENÇAS MÉDIAS ABSOLUTAS)
# ==========================

nemenyi_matrix = np.abs(pivot.values[:, :, None] - pivot.values[:, None, :]).mean(axis=0)
nemenyi_df = pd.DataFrame(nemenyi_matrix, index=pivot.columns, columns=pivot.columns)
#ordena
nemenyi_df = nemenyi_df.loc[ordem_modelos, ordem_modelos]
print("\n🧠 Matriz de Diferenças Médias - Nemenyi:\n", nemenyi_df)

# ==========================
# SALVAR RESULTADOS
# ==========================

# Gerar nomes dos arquivos
nome_csv = f'006_matriz_nemenyi_{metrica.lower()}.csv'
nome_png = f'006_heatmap_nemenyi_{metrica.lower()}.png'

# Salvar a matriz em CSV
nemenyi_df.to_csv(nome_csv)
print(f"\n💾 Matriz de Nemenyi salva em: {nome_csv}")

# ==========================
# HEATMAP
# ==========================

plt.figure(figsize=(10, 8))
sns.heatmap(nemenyi_df, annot=True, fmt=".4f", cmap="coolwarm", square=True)
plt.title(f'Heatmap - Diferença Média Absoluta (Nemenyi) - {metrica}')
plt.tight_layout()

# Salvar heatmap como PNG
plt.savefig(nome_png, dpi=300)
print(f"🖼️ Heatmap salvo em: {nome_png}")

plt.show()

# ==========================
# FIM
# ==========================
