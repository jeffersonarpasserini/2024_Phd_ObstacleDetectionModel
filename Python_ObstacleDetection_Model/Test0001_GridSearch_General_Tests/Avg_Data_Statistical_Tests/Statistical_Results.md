# Análise Estatística dos Modelos de Classificação Aplicados à Detecção de Obstáculos

## 1. Introdução

Nesta análise, foi aplicada uma metodologia estatística robusta para avaliar e ranquear o desempenho de diferentes modelos de classificação utilizados na tarefa de detecção de obstáculos para auxílio a deficientes visuais. A análise baseia-se em testes não paramétricos apropriados para cenários com múltiplas execuções e distribuições possivelmente não normais dos resultados.

---

## 2. Teste de Normalidade Global

O teste de normalidade global aplicado à métrica de **Acurácia (Accuracy)** retornou os seguintes valores:

- **Estatística**: 355396.8929  
- **Valor p**: < 0.0000

O resultado indica que a distribuição das acurácias **não segue uma distribuição normal**, justificando o uso de testes estatísticos não paramétricos como o de **Friedman** e o pós-teste de **Nemenyi**.

---

## 3. Teste de Friedman

O teste de Friedman foi aplicado às execuções dos modelos com base em suas acurácias. O teste revelou uma **diferença estatisticamente significativa** entre os modelos testados:

- **Estatística de Friedman (X²)**: 45665.0000  
- **Valor p**: < 0.0000

Isso indica que pelo menos um dos modelos apresenta desempenho significativamente diferente dos demais.

---

## 4. Pós-teste de Nemenyi

Após a detecção de diferenças estatísticas pelo teste de Friedman, o pós-teste de Nemenyi foi utilizado para realizar comparações pareadas entre os modelos. O resultado permitiu identificar pares de modelos cuja diferença de desempenho é **estatisticamente significativa**, com base nas acurácias medianas.

---

## 5. Top 10 Modelos com Melhor Desempenho

A tabela abaixo apresenta os **10 modelos com melhor desempenho**, ranqueados com base na métrica de acurácia ajustada (mediana) e no ranking médio de Friedman:

| Rank | Model Identifier                            | Accuracy | Friedman Rank |
|------|---------------------------------------------|----------|----------------|
| 1    | relu-avg-0.1-0.0001-2-256-adam               | 0.93600  | 1.0            |
| 2    | relu-avg-0.1-0.0001-2-512-adam               | 0.93500  | 2.0            |
| 3    | relu-max-0.1-0.0001-2-256-adam               | 0.93300  | 3.0            |
| 4    | relu-avg-0.1-0.0001-1-256-adam               | 0.93200  | 4.0            |
| 5    | relu-max-0.1-0.0001-2-512-adam               | 0.93100  | 5.0            |
| 6    | relu-avg-0.1-0.0001-2-256-sgd                | 0.93000  | 6.0            |
| 7    | relu-max-0.1-0.0001-1-256-adam               | 0.92900  | 7.0            |
| 8    | relu-avg-0.1-0.0001-2-512-sgd                | 0.92800  | 8.0            |
| 9    | relu-avg-0.1-0.0001-1-512-adam               | 0.92700  | 9.0            |
| 10   | relu-max-0.1-0.0001-2-256-sgd                | 0.92600  | 10.0           |

---

## 6. Conclusão

A análise confirmou que há **diferenças estatisticamente significativas** entre os modelos de classificação testados. A acurácia global dos modelos não apresentou distribuição normal, sendo apropriado o uso de testes não paramétricos. O uso do teste de Friedman e do pós-teste de Nemenyi permitiu ranquear de forma robusta os modelos, identificando aqueles com melhor desempenho estatístico em termos de detecção de obstáculos.

Esta abordagem estatística poderá ser utilizada como referência para avaliação e validação de novos modelos, bem como para justificar a escolha de algoritmos em estudos científicos e implementações práticas.

---

