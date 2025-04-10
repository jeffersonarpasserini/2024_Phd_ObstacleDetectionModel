
# Statistical Analysis of Classification Models for Obstacle Detection

This report presents the statistical analysis conducted on various classification models trained for obstacle detection, targeting applications for assisting visually impaired individuals.

## Normality Test

A global normality test (D’Agostino and Pearson’s test) was conducted to determine whether the Accuracy values of the evaluated models followed a normal distribution. The test indicated a non-normal distribution:

- **Statistic**: 10023.0831
- **p-value**: 0.00000000

This result suggests that non-parametric methods are more suitable for subsequent analysis.

## Friedman Test

To statistically compare the classification performance of the models, the Friedman Test was applied, which is appropriate for comparing multiple algorithms over several datasets or repetitions when normality cannot be assumed.

- **Friedman Chi-square**: 6047.00000000
- **p-value**: 0.49758156

Since the p-value is greater than 0.05, we **fail to reject** the null hypothesis. This means that no statistically significant difference was found between the models. However, ranking was still performed to assess relative performance.

## Implications

Although the statistical test does not indicate a significant difference, the practical application — particularly in safety-critical scenarios like obstacle detection — still benefits from ranking models by performance metrics. Therefore, weighted scoring and ranking were used to guide model selection based on performance.

## Weighted Scoring Justification

Weights were applied to the metrics as follows:

- **Recall (30%)** – Critical for detecting obstacles and minimizing false negatives.
- **F1-Score (25%)** – Balances recall and precision, relevant for reliability.
- **Accuracy (15%)** – Lower weight due to potential bias from class imbalance.
- **Precision (15%)** – Less critical than recall in this use case.
- **ROC-AUC (15%)** – Reflects model separability, but secondary to recall.

## Top 10 Ranked Models

|   Model | ExtractModel   | Pooling   | Parameters                                                                                                                                         |   Accuracy |   Precision |   Recall |   F1-Score |   ROC-AUC |   Weighted Score |   Friedman Rank |
|--------:|:---------------|:----------|:--------------------------------------------------------------------------------------------------------------------------------------------------|-----------:|------------:|---------:|-----------:|----------:|-----------------:|----------------:|
|      80 | MobileNetV1    | avg       | {'activation': 'relu', 'dropout_rate': 0.1, 'learning_rate': 0.0001, 'n_layers': 3, 'n_neurons': 256, 'optimizer': 'rmsprop', 'pooling': 'avg'}  |   0.966321 |    0.961538 | 0.975610 |   0.968523 |  0.965705 |         0.968848 |             1   |
|     131 | MobileNetV1    | avg       | {'activation': 'relu', 'dropout_rate': 0.2, 'learning_rate': 0.0005, 'n_layers': 2, 'n_neurons': 256, 'optimizer': 'adam', 'pooling': 'avg'}     |   0.963731 |    0.966825 | 0.966825 |   0.966825 |  0.963412 |         0.965849 |             3.5 |
|     265 | MobileNetV1    | avg       | {'activation': 'relu', 'dropout_rate': 0.4, 'learning_rate': 0.0001, 'n_layers': 2, 'n_neurons': 128, 'optimizer': 'adam', 'pooling': 'avg'}     |   0.963731 |    0.966825 | 0.966825 |   0.966825 |  0.963412 |         0.965849 |             3.5 |
|      15 | MobileNetV1    | avg       | {'activation': 'relu', 'dropout_rate': 0.0, 'learning_rate': 0.0001, 'n_layers': 2, 'n_neurons': 512, 'optimizer': 'adam', 'pooling': 'avg'}     |   0.963731 |    0.966825 | 0.966825 |   0.966825 |  0.963412 |         0.965849 |             3.5 |
|     264 | MobileNetV1    | avg       | {'activation': 'relu', 'dropout_rate': 0.4, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam', 'pooling': 'avg'}     |   0.963731 |    0.966825 | 0.966825 |   0.966825 |  0.963412 |         0.965849 |             3.5 |
|     273 | MobileNetV1    | avg       | {'activation': 'relu', 'dropout_rate': 0.4, 'learning_rate': 0.005, 'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam', 'pooling': 'avg'}      |   0.961140 |    0.971154 | 0.957346 |   0.964200 |  0.961530 |         0.962327 |            29.5 |
|      11 | MobileNetV1    | avg       | {'activation': 'relu', 'dropout_rate': 0.0, 'learning_rate': 0.0001, 'n_layers': 3, 'n_neurons': 32, 'optimizer': 'adam', 'pooling': 'avg'}      |   0.961140 |    0.953704 | 0.976303 |   0.964871 |  0.959580 |         0.965272 |            29.5 |
|      74 | MobileNetV1    | avg       | {'activation': 'relu', 'dropout_rate': 0.1, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 256, 'optimizer': 'adam', 'pooling': 'avg'}     |   0.961140 |    0.966667 | 0.962085 |   0.964371 |  0.961043 |         0.963046 |            29.5 |
|     132 | MobileNetV1    | avg       | {'activation': 'relu', 'dropout_rate': 0.2, 'learning_rate': 0.0005, 'n_layers': 2, 'n_neurons': 512, 'optimizer': 'adam', 'pooling': 'avg'}     |   0.961140 |    0.966667 | 0.962085 |   0.964371 |  0.961043 |         0.963046 |            29.5 |
|     223 | MobileNetV1    | avg       | {'activation': 'relu', 'dropout_rate': 0.3, 'learning_rate': 0.001, 'n_layers': 3, 'n_neurons': 128, 'optimizer': 'adam', 'pooling': 'avg'}     |   0.961140 |    0.962264 | 0.966825 |   0.964539 |  0.960555 |         0.963776 |            29.5 |

## Conclusion

Even though statistical significance was not observed, the ranking process enables the identification of the best-performing models. For practical applications, especially those involving safety, it's crucial to prioritize metrics such as recall and F1-score to reduce false negatives. The top-ranked models identified in this analysis can guide future deployment and refinement strategies.
