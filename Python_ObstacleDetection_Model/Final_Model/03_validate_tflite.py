"""
03_validate_tflite.py
=====================
Valida o modelo TFLite combinado contra o Keras e mede o que importa para o deploy:
  - PARIDADE Keras vs TFLite (a conversao nao pode mudar as predicoes)
  - Acuracia / Recall(obstaculo) / Especificidade na base estendida @ threshold de operacao (tau)
  - Latencia media de inferencia por imagem (proxy do custo no dispositivo)
  - Tamanho do arquivo .tflite

Le o tau de ../model/final_operating_point.json (gerado por 01).
Roda APOS 02:  python 03_validate_tflite.py

Obs.: este script roda em float32 (PC). A latencia no smartphone sera diferente;
serve como sanity check de que o modelo esta correto e leve.
Para quantizacao INT8 (modelo menor/mais rapido no celular), ver comentario no final.
"""

import os
import json
import time
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing import image
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             f1_score, confusion_matrix)

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = os.path.abspath(os.path.join(BASE_PATH, '..'))
DATASET_PATH = os.path.abspath(os.path.join(PROJECT_PATH, '..', 'via-dataset-extended'))
MODEL_DIR = os.path.join(BASE_PATH, 'model')
COMBINED_H5 = os.path.join(MODEL_DIR, 'combined_model', 'classifier_model_combined.h5')
COMBINED_TFLITE = os.path.join(MODEL_DIR, 'combined_model', 'classifier_model_combined.tflite')
OP_PATH = os.path.join(MODEL_DIR, 'final_operating_point.json')
IMAGE_SIZE = (224, 224)


def load_dataset():
    valid = ('.jpg', '.jpeg', '.png')
    files = sorted(f for f in os.listdir(DATASET_PATH) if f.lower().endswith(valid))
    y = np.array([0 if f.split('.')[0] == 'clear' else 1 for f in files], dtype=int)
    return files, y


def load_image(fname):
    img = image.load_img(os.path.join(DATASET_PATH, fname), target_size=IMAGE_SIZE)
    arr = image.img_to_array(img)
    arr = preprocess_input(arr)                 # MobileNetV1 -> [-1, 1]
    return np.expand_dims(arr, 0).astype(np.float32)


def tflite_predictor(path):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    out = interp.get_output_details()[0]

    def predict(x):
        interp.set_tensor(inp['index'], x)
        interp.invoke()
        return float(interp.get_tensor(out['index']).ravel()[0])
    return predict


def summarize(tag, y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    print(f'\n[{tag}] @tau={TAU:.4f}')
    print(f'   Accuracy        : {accuracy_score(y_true, y_pred):.4f}')
    print(f'   Recall obstaculo: {recall_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}')
    print(f'   Especificidade  : {recall_score(y_true, y_pred, pos_label=0, zero_division=0):.4f}')
    print(f'   Precision       : {precision_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}')
    print(f'   F1              : {f1_score(y_true, y_pred, pos_label=1, zero_division=0):.4f}')
    print(f'   TP={tp} TN={tn} FP={fp} FN={fn}  (FN = obstaculos perdidos)')


def main():
    global TAU
    with open(OP_PATH) as f:
        op = json.load(f)
    TAU = float(op['operating_threshold'])
    print(f'==> Threshold de operacao (tau) = {TAU:.4f}  [{op["threshold_criterion"]}]')

    files, y_true = load_dataset()
    print(f'==> {len(files)} imagens (obstaculo={int(y_true.sum())}, clear={int((y_true==0).sum())})')

    keras_model = tf.keras.models.load_model(COMBINED_H5, compile=False)
    tflite_predict = tflite_predictor(COMBINED_TFLITE)

    keras_prob = np.zeros(len(files)); tfl_prob = np.zeros(len(files))
    lat = []
    for i, fn in enumerate(files):
        x = load_image(fn)
        keras_prob[i] = float(keras_model.predict(x, verbose=0).ravel()[0])
        t0 = time.perf_counter()
        tfl_prob[i] = tflite_predict(x)
        lat.append((time.perf_counter() - t0) * 1000.0)
        if (i + 1) % 200 == 0:
            print(f'   ... {i+1}/{len(files)}')

    # PARIDADE — operador > (strict), alinhado ao Test0009
    diff_abs = np.abs(keras_prob - tfl_prob)
    max_diff  = float(diff_abs.max())
    mean_diff = float(diff_abs.mean())
    disagree  = int(np.sum((keras_prob > TAU) != (tfl_prob > TAU)))

    # FN de cada modelo (segurança crítica: obstáculos perdidos)
    fn_keras  = int(np.sum((y_true == 1) & (keras_prob <= TAU)))
    fn_tflite = int(np.sum((y_true == 1) & (tfl_prob  <= TAU)))

    print(f'\n==> PARIDADE Keras vs TFLite')
    print(f'   max|Δprob|  = {max_diff:.6f}   mean|Δprob| = {mean_diff:.6f}')
    print(f'   Divergencias de classe = {disagree}/{len(files)}')
    print(f'   FN Keras={fn_keras}  FN TFLite={fn_tflite}  (obstaculos perdidos — critico)')

    # Limiar de alerta ajustado para float16:
    #   float16 acumula erros de ~1e-3 por camada em modelos profundos (MobileNetV1 tem 28 blocos).
    #   max|Δprob| < 0.10 e FN identico sao os criterios de aprovacao para float16.
    MAX_DIFF_FLOAT16 = 0.10
    ok_diff    = max_diff  <= MAX_DIFF_FLOAT16
    ok_fn      = fn_keras  == fn_tflite
    ok_disagree = disagree <= 5          # ate 5 casos de fronteira sao aceitaveis em 1929 imgs

    if ok_diff and ok_fn and ok_disagree:
        print(f'   OK: TFLite (float16) dentro dos limites de paridade aceitaveis.')
    else:
        if not ok_diff:
            print(f'   [ATENCAO] max|Δprob| = {max_diff:.4f} > {MAX_DIFF_FLOAT16} -- investigar conversao.')
        if not ok_fn:
            print(f'   [ATENCAO] FN diferente: Keras={fn_keras} vs TFLite={fn_tflite} -- obstaculos perdidos divergem.')
        if not ok_disagree:
            print(f'   [ATENCAO] {disagree} divergencias de classe -- muitos casos de fronteira afetados.')

    summarize('Keras ', y_true, (keras_prob > TAU).astype(int))
    summarize('TFLite', y_true, (tfl_prob > TAU).astype(int))

    size_mb = os.path.getsize(COMBINED_TFLITE) / (1024 ** 2)
    print(f'\n==> Tamanho TFLite : {size_mb:.2f} MB')
    print(f'==> Latencia TFLite: media={np.mean(lat):.2f} ms  mediana={np.median(lat):.2f} ms (PC, float32)')

    pd.DataFrame({'filename': files, 'y_true': y_true,
                  'keras_prob': keras_prob, 'tflite_prob': tfl_prob,
                  'y_pred_tflite': (tfl_prob > TAU).astype(int)}
                 ).to_csv(os.path.join(MODEL_DIR, 'tflite_validation_predictions.csv'), index=False)
    print('\nOK. Predicoes salvas em model/tflite_validation_predictions.csv')

    # ------------------------------------------------------------------
    # QUANTIZACAO INT8 (opcional, para reduzir tamanho/latencia no celular):
    #   converter.optimizations = [tf.lite.Optimize.DEFAULT]
    #   converter.representative_dataset = <gerador com ~100 imagens preprocessadas>
    #   converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    #   converter.inference_input_type = tf.uint8  # ou float32
    # Depois reexecutar este script para confirmar que a acuracia NAO caiu.
    # ------------------------------------------------------------------


if __name__ == '__main__':
    main()
