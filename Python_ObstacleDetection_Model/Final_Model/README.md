# Final_Model — modelo fechado para o protótipo mobile

Modelo escolhido: **Model03 / F2 / recall-first** (ver decisão e justificativa na memória do projeto).
Extrator **MobileNetV1** (avg, congelado) + **MLP** (`relu`, dropout 0.4, lr 0.001, 1 camada / 256 neurônios,
`adam`, perda `get_f1_loss(alpha=0.6, beta=1.5)`). Polaridade: **obstáculo = 1**, clear = 0.
A saída do modelo é **P(obstáculo)**; o app aplica o threshold de operação (`tau`).

## Ordem de execução (na máquina com GPU + TensorFlow)

```bash
python 01_train_final_model.py      # treina + serializa MLP + calibra tau (F2) -> model/
python 02_build_combined_tflite.py  # funde MobileNetV1+MLP e exporta .tflite
python 03_validate_tflite.py        # paridade Keras vs TFLite, acurácia@tau, latência, tamanho
```

## Artefatos gerados (em `../model/`)

| Arquivo | Conteúdo |
|---|---|
| `classifier_model/00_classifier_model.h5` | MLP final (treinado em 100% da base) |
| `final_operating_point.json` | config + `tau` (threshold) + métricas de calibração |
| `threshold_sweep_calib.csv` | curva recall/especificidade/F2 por threshold (para a tese) |
| `combined_model/classifier_model_combined.h5` | modelo end-to-end imagem→P(obstáculo) |
| `combined_model/classifier_model_combined.tflite` | **modelo para o Android** |
| `tflite_validation_predictions.csv` | predições de validação Keras vs TFLite |

## Notas

- O `tau` é calibrado por **F2** (recall pesa 2×) num holdout estratificado de 20%.
  Pontos de operação alternativos (0.3/0.4/0.5) ficam no JSON para a tese.
- Recall de obstáculo satura em ~0.95 → o aparelho é **complemento da bengala**.
- Para INT8 (modelo menor/mais rápido no celular), ver o bloco comentado no fim de `03`.
- Supera o `../combined_model.py` legado (estava em MobileNetV2 — incompatível com o extrator V1).
- No app Kotlin, replicar **exatamente** o `preprocess_input` do MobileNetV1 (escala para [-1, 1]).
