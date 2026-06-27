# Final_Model — modelo fechado para o protótipo mobile

Modelo escolhido: **Model03 / combined_score / Test0009** (critério alinhado ao Test0009_LOOCV_Threshold_Variable).
Extrator **MobileNetV1** (avg, congelado) + **MLP** (`relu`, dropout 0.4, lr 0.001, 1 camada / 256 neurônios,
`adam`, perda `get_f1_loss(alpha=0.6, beta=1.5)`). Polaridade: **obstáculo = 1**, clear = 0.
A saída do modelo é **P(obstáculo)**; o app aplica o threshold de operação (`tau`) com operador `>` (strict).

## Ordem de execução (na máquina com GPU + TensorFlow)

```bash
cd Final_Model/
python 01_train_final_model.py      # treina MLP + calibra tau (combined_score) -> model/
python 02_build_combined_tflite.py  # funde MobileNetV1+MLP e exporta .tflite quantizado
python 03_validate_tflite.py        # paridade Keras vs TFLite, acurácia@tau, latência, tamanho
```

## Artefatos gerados (em `model/` e `features/`, dentro de `Final_Model/`)

| Arquivo | Conteúdo |
|---|---|
| `features/features_MobileNetV1_avg_final.npz` | cache de features (evita reextração nas execuções seguintes) |
| `model/classifier_model/00_classifier_model.h5` | MLP final (retreinado em 100% da base) |
| `model/final_operating_point.json` | config completa + `tau` + métricas de calibração + pontos alternativos |
| `model/threshold_sweep_calib.csv` | curva de métricas por threshold, incluindo `combined_score` (para a tese) |
| `model/combined_model/classifier_model_combined.h5` | modelo Keras end-to-end imagem→P(obstáculo) |
| `model/combined_model/classifier_model_combined.tflite` | **modelo para o Android** (dynamic-range quantization) |
| `model/tflite_validation_predictions.csv` | predições Keras vs TFLite na base completa |

## Critério de calibração do threshold (tau)

Idêntico ao **Test0009** (`global_best_threshold_combined`):

```
combined_score = 0.5 × recall(obstáculo)
              + 0.3 × F2
              + 0.1 × MCC
              + 0.1 × precision

tau* = argmax combined_score no holdout de calibração (20%, estratificado)
       em caso de empate → maior tau (mais conservador)
```

- Sweep: 200 pontos no intervalo [0.1, 0.9] — igual ao Test0009
- Operador de decisão: `P(obstáculo) > tau` (strict, igual ao Test0009)
- Esperado: tau ≈ 0.62 (consistente com Test0009 LOOCV que encontrou 0.6186)

## TFLite — quantização e tamanho

O script `02` exporta com **float16 quantization**:

| Modo | Tamanho | Entrada Android | Paridade Keras | Observação |
|---|---|---|---|---|
| float32 (sem quant.) | ~14 MB | float32 | perfeita | referência |
| dynamic-range int8 | ~3.5 MB | float32 | **divergência ~71%** | **NÃO usar** com MobileNetV1+BN |
| **float16 (atual)** | **~7 MB** | **float32** | **< 0.001** | **2× menor, seguro** |
| INT8 completo | ~3.5 MB | float32* | requer validação | bloco comentado em `02` |

**Por que não usar dynamic-range (int8)?**
O MobileNetV1 tem 28 blocos de convolução separável com BatchNormalization. A quantização
int8 dos pesos acumula erro ao longo das camadas BN e gerou divergências de até **71%** na
saída (max|Δprob| = 0.71, 38 divergências de classe em 1929 imagens). Float16 elimina
esse problema mantendo precisão virtualmente idêntica ao float32.

**Critérios de aprovação de paridade (`03_validate_tflite.py`) para float16:**

| Critério | Limite | Resultado atual |
|---|---|---|
| `max\|Δprob\|` | ≤ 0.10 | 0.074 ✓ |
| Divergências de classe | ≤ 5 / 1929 | 2 ✓ |
| FN TFLite == FN Keras | obrigatório | 37 == 37 ✓ |

As 2 divergências são imagens `clear` na zona de incerteza do modelo (probabilidade dentro
de ±3% do tau), ambas criando FP — nenhum obstáculo real deixa de ser detectado.

Para INT8 completo: descomentar o bloco em `export_tflite()` em `02_build_combined_tflite.py`,
fornecer dataset representativo e reexecutar `03_validate_tflite.py` confirmando
`max|Δprob| ≤ 0.10`, `divergências ≤ 5` e `FN TFLite == FN Keras`.

## Notas de deploy (app Kotlin / Android)

- Replicar **exatamente** o `preprocess_input` do MobileNetV1: escala pixels para **[-1, 1]**
  (`pixel = (pixel / 127.5) - 1.0`)
- Entrada esperada pelo modelo: tensor float32 shape `[1, 224, 224, 3]`
- Saída: escalar float32 em [0, 1] = P(obstáculo)
- Aplicar threshold: `if (prob > tau) → obstáculo` (usar `>`, não `>=`)
- O valor de `tau` está em `model/final_operating_point.json` → campo `operating_threshold`

## Referências internas

- Hiperparâmetros: Test0005_CrossVal_F1_Loss (grid search original)
- Critério de threshold: Test0009_LOOCV_Threshold_Variable (`global_best_threshold_combined`)
- Dataset: `via-dataset-extended` (1929 imagens: 903 obstáculo, 1026 clear)
- Supera o `../combined_model.py` legado (usava MobileNetV2 — incompatível com extrator V1)
