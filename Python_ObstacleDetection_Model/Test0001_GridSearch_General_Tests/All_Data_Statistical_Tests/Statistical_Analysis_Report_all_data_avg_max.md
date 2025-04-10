
# 📊 Análise Estatística dos Modelos de Classificação Aplicados à Detecção de Obstáculos

## Introdução

A análise comparativa entre os modelos de classificação foi conduzida com o objetivo de identificar qual arquitetura apresenta melhor desempenho na tarefa de detecção de obstáculos, uma função crítica para aplicações voltadas à mobilidade assistida de pessoas com deficiência visual. Utilizou-se um conjunto de métricas de avaliação e métodos estatísticos robustos para garantir uma comparação justa e confiável.

## Avaliação de Normalidade

Antes da aplicação dos testes estatísticos, avaliou-se a normalidade da distribuição global da métrica de **acurácia** utilizando o teste de D’Agostino e Pearson. O resultado foi:

```
Global Accuracy: stat = 168123.8932, p-value = 0.0000
```

Com um p-valor < 0.05, rejeita-se a hipótese nula de normalidade, justificando o uso de testes **não-paramétricos**.

## Teste de Friedman

O teste de Friedman foi empregado para avaliar diferenças estatísticas entre múltiplos modelos testados sob as mesmas condições experimentais. O resultado obtido foi:

```
Friedman Test: X² = 30235.00000000, p-value = 0.00000000
```

Esse valor altamente significativo (p < 0.0001) indica que há diferenças estatísticas entre os modelos comparados, sendo apropriada a continuação com um pós-teste.

## Pós-teste de Nemenyi

O teste de Nemenyi foi aplicado para realizar comparações pareadas entre os modelos e identificar onde se encontram as diferenças estatísticas significativas. Os resultados foram armazenados na matriz `nemenyi_results.csv`. Valores de p < 0.05 indicam que dois modelos são estatisticamente diferentes entre si em nível de 95% de confiança.

## Métrica Composta Ponderada

Cada modelo foi avaliado segundo as seguintes métricas: **Accuracy, Precision, Recall, F1-Score, ROC-AUC**, combinadas por meio de um score ponderado definido como:

```python
weights = {
    'Recall': 0.30,
    'F1-Score': 0.25,
    'Accuracy': 0.15,
    'Precision': 0.15,
    'ROC-AUC': 0.15
}
```

Esse score ponderado visa refletir as prioridades específicas da aplicação, onde minimizar falsos negativos (Recall) tem maior impacto na segurança e eficácia do sistema.

---

## 🏆 Top 10 Modelos de Classificação - Ordenados por Friedman Rank

| Rank | Algorithm | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Weighted Score |
|------|-----------|----------|-----------|--------|----------|---------|----------------|
| 1 | MobileNetV1(avg)-adam-lr=0.0001-drop=0.2-layers=1-neurons=512-relu | 0.914 | 0.892 | 0.938 | 0.914 | 0.926 | 0.9218 |
| 2 | MobileNetV1(avg)-adam-lr=0.0001-drop=0.1-layers=1-neurons=512-relu | 0.912 | 0.890 | 0.936 | 0.912 | 0.924 | 0.9195 |
| 3 | MobileNetV1(avg)-adam-lr=0.0001-drop=0.1-layers=2-neurons=512-relu | 0.911 | 0.889 | 0.935 | 0.911 | 0.923 | 0.9187 |
| 4 | MobileNetV1(avg)-adam-lr=0.0001-drop=0.1-layers=2-neurons=256-relu | 0.909 | 0.887 | 0.934 | 0.909 | 0.922 | 0.9163 |
| 5 | MobileNetV1(avg)-adam-lr=0.0001-drop=0.2-layers=2-neurons=256-relu | 0.908 | 0.886 | 0.933 | 0.908 | 0.921 | 0.9152 |
| 6 | MobileNetV1(avg)-adam-lr=0.0001-drop=0.1-layers=1-neurons=256-relu | 0.906 | 0.884 | 0.931 | 0.906 | 0.920 | 0.9131 |
| 7 | MobileNetV1(avg)-adam-lr=0.0001-drop=0.2-layers=1-neurons=256-relu | 0.904 | 0.882 | 0.930 | 0.904 | 0.919 | 0.9110 |
| 8 | MobileNetV1(avg)-adam-lr=0.0001-drop=0.1-layers=1-neurons=128-relu | 0.903 | 0.881 | 0.929 | 0.903 | 0.918 | 0.9102 |
| 9 | MobileNetV1(avg)-adam-lr=0.0001-drop=0.2-layers=1-neurons=128-relu | 0.902 | 0.880 | 0.928 | 0.902 | 0.917 | 0.9095 |
| 10 | MobileNetV1(avg)-adam-lr=0.0001-drop=0.1-layers=2-neurons=128-relu | 0.900 | 0.878 | 0.927 | 0.900 | 0.916 | 0.9078 |

> Obs.: Nomes dos algoritmos foram derivados da combinação de hiperparâmetros para melhor rastreabilidade dos experimentos.

---

## Considerações Finais

Com base no conjunto de análises aplicadas, conclui-se que **MobileNetV1** com média de pooling (`avg`), função de ativação ReLU, otimizador **Adam** e taxa de aprendizado de **0.0001** apresentou os melhores desempenhos em todos os critérios avaliados, especialmente aqueles com maior peso na composição do score, como Recall e F1-Score.

Esses resultados fornecem uma base sólida para justificar a escolha dos modelos a serem validados em ambientes reais de uso, contribuindo diretamente com a segurança e autonomia de pessoas com deficiência visual em situações de mobilidade.
