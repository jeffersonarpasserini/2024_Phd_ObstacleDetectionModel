"""
01_train_final_model.py
========================
Treina o MODELO FINAL fechado para o protótipo mobile e o serializa.

Decisão: Model03 / F2 / recall-first (ver memoria do projeto: final-model-decision).
Receita 100% fiel ao Test0009 (get_f1_loss alpha=0.6, beta=1.5).

Pipeline:
  1) Extrai features MobileNetV1 (avg pooling) da base estendida (com cache proprio,
     alinhado: features + labels + filenames).
  2) Split estratificado treino/calibracao (80/20) -> treina MLP -> varre threshold e
     escolhe o tau* que MAXIMIZA o combined_score (0.5*recall+0.3*F2+0.1*MCC+0.1*precision)
     no conjunto de calibracao — criterio identico ao Test0009 (global_best_threshold_combined).
  3) RETREINA o MLP em 100% dos dados (mesma config) -> pesos finais para deploy.
  4) Salva:
       model/classifier_model/00_classifier_model.h5   (MLP final)
       model/final_operating_point.json                (config + tau* + metricas calib)
       model/threshold_sweep_calib.csv                 (curva completa para a tese)

IMPORTANTE: polaridade fixada -> obstaculo (non-clear) = 1, clear = 0.
            O app aplica o threshold sobre P(obstaculo).

Rodar:  python 01_train_final_model.py
Requer: tensorflow, scikit-learn, numpy, pandas, pillow  (na maquina com GPU).
"""

import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_score, recall_score, f1_score, fbeta_score,
                             accuracy_score, confusion_matrix, matthews_corrcoef)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

# ----------------------------------------------------------------------------
# Reprodutibilidade (igual aos experimentos)
# ----------------------------------------------------------------------------
os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# ----------------------------------------------------------------------------
# Caminhos
# ----------------------------------------------------------------------------
BASE_PATH = os.path.dirname(os.path.abspath(__file__))                 # .../Final_Model
PROJECT_PATH = os.path.abspath(os.path.join(BASE_PATH, '..'))          # .../Python_ObstacleDetection_Model
DATASET_PATH = os.path.abspath(os.path.join(PROJECT_PATH, '..', 'via-dataset-extended'))

MODEL_DIR = os.path.join(BASE_PATH, 'model')
CLASSIFIER_DIR = os.path.join(MODEL_DIR, 'classifier_model')
FEATURE_DIR = os.path.join(BASE_PATH, 'features')
for d in (MODEL_DIR, CLASSIFIER_DIR, FEATURE_DIR):
    os.makedirs(d, exist_ok=True)

# ----------------------------------------------------------------------------
# Hiperparametros do MODELO FINAL (Model03 / Test0009) -- NAO ALTERAR sem motivo
# ----------------------------------------------------------------------------
MODEL_TYPE = 'MobileNetV1'
POOLING = 'avg'
ALPHA = 1.0
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3

MLP = dict(activation='relu', dropout_rate=0.4, learning_rate=0.001,
           n_layers=1, n_neurons=256, optimizer='adam',
           f1_alpha=0.6, f1_beta=1.5)
TRAIN = dict(epochs=250, batch_size=32, earlystop_patience=50,
             reduceLR_factor=0.5, reduceLR_patience=10, val_split=0.2)

CALIB_FRACTION = 0.20   # holdout para calibrar o threshold (estratificado)
FBETA = 2.0             # F2: recall pesa 2x (usado no combined_score do Test0009)

# Pesos do combined_score (replicados do Test0009 global_best_threshold_combined)
COMBINED_W = dict(recall=0.5, f2=0.3, mcc=0.1, precision=0.1)


# ----------------------------------------------------------------------------
# Perda e metrica (copiadas IDENTICAS do Test0009)
# ----------------------------------------------------------------------------
def f1_metric(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.round(K.clip(y_pred, 0, 1))
    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())
    return 2 * (precision * recall) / (precision + recall + K.epsilon())


def get_f1_loss(alpha=0.6, beta=1.0):
    def f1_loss(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        tp = K.sum(y_true * y_pred)
        fp = K.sum((1 - y_true) * y_pred)
        fn = K.sum(y_true * (1 - y_pred))
        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())
        f1 = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + K.epsilon())
        return 1 - (alpha * f1 + (1 - alpha) * precision)
    return f1_loss


# ----------------------------------------------------------------------------
# Dados / extracao de features (obstaculo=1, clear=0)
# ----------------------------------------------------------------------------
def load_data():
    valid = ('.jpg', '.jpeg', '.png')
    filenames = sorted(f for f in os.listdir(DATASET_PATH) if f.lower().endswith(valid))
    # obstaculo (non-clear) = 1 ; clear = 0
    categories = [0 if f.split('.')[0] == 'clear' else 1 for f in filenames]
    return pd.DataFrame({'filename': filenames, 'category': categories})


def get_extract_model():
    from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
    model = MobileNet(weights='imagenet', include_top=False, pooling=POOLING,
                      input_shape=IMAGE_SIZE + (IMAGE_CHANNELS,), alpha=ALPHA)
    return model, preprocess_input


def extract_features(df):
    """Extrai features (sem augmentation) e devolve (features, labels, filenames) alinhados."""
    cache = os.path.join(FEATURE_DIR, f'features_{MODEL_TYPE}_{POOLING}_final.npz')
    if os.path.exists(cache):
        print(f'[cache] Carregando features: {cache}')
        d = np.load(cache, allow_pickle=True)
        return d['features'], d['labels'], d['filenames']

    print('[extract] Extraindo features MobileNetV1 (avg) da base estendida...')
    model, preprocessing_function = get_extract_model()
    labels = df['category'].to_numpy().astype(int)
    df_str = df.copy()
    df_str['category'] = df_str['category'].replace({0: 'clear', 1: 'non-clear'})

    datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)
    batch_size = 4
    steps = int(np.ceil(df_str.shape[0] / batch_size))
    generator = datagen.flow_from_dataframe(
        df_str, DATASET_PATH, x_col='filename', y_col='category',
        class_mode='binary', target_size=IMAGE_SIZE, batch_size=batch_size, shuffle=False)
    features = model.predict(generator, steps=steps, verbose=1)

    filenames = df['filename'].to_numpy()
    np.savez_compressed(cache, features=features, labels=labels, filenames=filenames)
    print(f'[cache] Features salvas: {cache}  shape={features.shape}')
    return features, labels, filenames


# ----------------------------------------------------------------------------
# Classificador (build IDENTICO ao Test0009)
# ----------------------------------------------------------------------------
def build_mlp(input_shape):
    model = Sequential([Input(shape=(input_shape,))])
    for _ in range(MLP['n_layers']):
        model.add(Dense(MLP['n_neurons'], activation=MLP['activation']))
        if MLP['dropout_rate'] > 0:
            model.add(Dropout(MLP['dropout_rate']))
    model.add(Dense(1, activation='sigmoid'))
    opt = Adam(learning_rate=MLP['learning_rate']) if MLP['optimizer'] == 'adam' \
        else RMSprop(learning_rate=MLP['learning_rate'])
    loss_fn = get_f1_loss(alpha=MLP['f1_alpha'], beta=MLP['f1_beta'])
    model.compile(optimizer=opt, loss=loss_fn, metrics=['accuracy', f1_metric])
    return model


def fit_mlp(model, X, y):
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=TRAIN['earlystop_patience'],
                      restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=TRAIN['reduceLR_factor'],
                          patience=TRAIN['reduceLR_patience'], min_lr=1e-5),
    ]
    model.fit(X, y.astype('float32'),
              validation_split=TRAIN['val_split'],
              epochs=TRAIN['epochs'], batch_size=TRAIN['batch_size'],
              callbacks=callbacks, verbose=2)
    return model


# ----------------------------------------------------------------------------
# Calibracao do threshold por F2 (recall-first)
# ----------------------------------------------------------------------------
def sweep_threshold(y_true, y_prob):
    """Varre thresholds e calcula métricas — alinhado ao Test0009 (range 0.1-0.9, 200 pts, operador >)."""
    rows = []
    for thr in np.linspace(0.1, 0.9, 200):
        y_pred = (y_prob > thr).astype(int)
        rec  = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        f2   = fbeta_score(y_true, y_pred, beta=FBETA, pos_label=1, zero_division=0)
        mcc  = matthews_corrcoef(y_true, y_pred) if len(set(y_pred)) > 1 else 0.0
        prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rows.append(dict(
            threshold=round(float(thr), 4),
            recall_obstacle=rec,
            specificity=recall_score(y_true, y_pred, pos_label=0, zero_division=0),
            precision=prec,
            f1=f1_score(y_true, y_pred, pos_label=1, zero_division=0),
            f2=f2,
            accuracy=accuracy_score(y_true, y_pred),
            mcc=mcc,
            combined_score=COMBINED_W['recall']*rec + COMBINED_W['f2']*f2
                           + COMBINED_W['mcc']*mcc + COMBINED_W['precision']*prec,
        ))
    return pd.DataFrame(rows)


def metrics_at(y_true, y_prob, thr):
    y_pred = (y_prob > thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return dict(
        threshold=float(thr),
        accuracy=accuracy_score(y_true, y_pred),
        recall_obstacle=recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        specificity=recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        precision=precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        f1=f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        f2=fbeta_score(y_true, y_pred, beta=FBETA, pos_label=1, zero_division=0),
        mcc=matthews_corrcoef(y_true, y_pred) if len(set(y_pred)) > 1 else 0.0,
        TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn),
    )


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    print('==> Carregando dados e extraindo features (MobileNetV1 avg)')
    df = load_data()
    print(f'    {len(df)} imagens | obstaculo={int(df.category.sum())} clear={int((df.category==0).sum())}')
    X, y, _ = extract_features(df)
    X = np.asarray(X); y = np.asarray(y).astype(int)

    # ---- 1) split de calibracao (estratificado) ----
    X_tr, X_cal, y_tr, y_cal = train_test_split(
        X, y, test_size=CALIB_FRACTION, stratify=y, random_state=SEED)

    print('\n==> [Etapa A] Treinando MLP no split de treino para calibrar threshold')
    cal_model = build_mlp(X.shape[1])
    cal_model = fit_mlp(cal_model, X_tr, y_tr)
    y_cal_prob = cal_model.predict(X_cal, verbose=0).ravel()

    sweep = sweep_threshold(y_cal, y_cal_prob)
    sweep_path = os.path.join(MODEL_DIR, 'threshold_sweep_calib.csv')
    sweep.to_csv(sweep_path, index=False)

    # Critério Test0009: argmax combined_score; em empate escolhe o MAIOR tau (mais conservador)
    best_score = sweep['combined_score'].max()
    candidates = sweep[sweep['combined_score'] == best_score]
    tau = float(candidates['threshold'].max())
    print(f'\n==> tau* (combined_score-otimo, criterio Test0009) = {tau:.4f}')
    print('    Metricas no holdout de calibracao @ tau*:')
    calib_metrics = metrics_at(y_cal, y_cal_prob, tau)
    for k, v in calib_metrics.items():
        print(f'      {k}: {v}')

    # pontos de operacao alternativos para a tese
    alt = {f'thr_{t}': metrics_at(y_cal, y_cal_prob, t) for t in (0.3, 0.5, 0.6)}

    # ---- 2) retreino em 100% dos dados (pesos finais p/ deploy) ----
    print('\n==> [Etapa B] Retreinando MLP em 100% dos dados (pesos finais)')
    final_model = build_mlp(X.shape[1])
    final_model = fit_mlp(final_model, X, y)

    h5_path = os.path.join(CLASSIFIER_DIR, '00_classifier_model.h5')
    final_model.save(h5_path)
    print(f'    MLP final salvo: {h5_path}')

    # ---- 3) operating point + config ----
    op = dict(
        model_name='Model03_F2_recall_first',
        extractor=dict(type=MODEL_TYPE, pooling=POOLING, alpha=ALPHA,
                       image_size=list(IMAGE_SIZE), feature_dim=int(X.shape[1]),
                       preprocess='mobilenet_v1 preprocess_input -> [-1,1]'),
        classifier=MLP, training=TRAIN,
        positive_class='obstacle (non-clear) = 1 ; clear = 0',
        operating_threshold=tau,
        threshold_criterion=(
            f'argmax combined_score (0.5*recall + 0.3*F{int(FBETA)} + 0.1*MCC + 0.1*precision) '
            f'no holdout de calibracao ({int(CALIB_FRACTION*100)}%) — criterio Test0009; '
            f'empate resolvido pelo maior tau'
        ),
        combined_weights=COMBINED_W,
        calib_metrics=calib_metrics,
        alt_operating_points=alt,
        seed=SEED,
    )
    op_path = os.path.join(MODEL_DIR, 'final_operating_point.json')
    with open(op_path, 'w') as f:
        json.dump(op, f, indent=2)
    print(f'    Ponto de operacao salvo: {op_path}')
    print('\nOK. Proximo passo: 02_build_combined_tflite.py')


if __name__ == '__main__':
    main()
