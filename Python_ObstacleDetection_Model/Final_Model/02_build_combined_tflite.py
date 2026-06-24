"""
02_build_combined_tflite.py
===========================
Funde MobileNetV1 (extrator, congelado) + MLP final num UNICO modelo Keras
end-to-end (imagem 224x224x3 -> P(obstaculo)) e exporta para TensorFlow Lite.

Corrige os problemas do combined_model.py legado:
  - usa MobileNetV1 (avg pooling), nao V2  -> dimensao de feature compativel (1024)
  - sem Flatten redundante (avg pooling ja entrega vetor 1D)
  - carrega o classificador final treinado por 01_train_final_model.py
  - mantem a polaridade: saida = P(obstaculo), obstaculo=1

Saidas (em ../model/combined_model/):
  classifier_model_combined.h5
  classifier_model_combined.tflite
  model_combined_architecture.png   (se pydot/graphviz disponiveis)

Rodar APOS 01:  python 02_build_combined_tflite.py
"""

import os
import tensorflow as tf
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
from tensorflow.keras import backend as K

# Mesma perda usada no treino -> necessaria para load_model do MLP
def get_f1_loss(alpha=0.6, beta=1.5):
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

def f1_metric(y_true, y_pred):
    y_true = K.cast(y_true, 'float32')
    y_pred = K.round(K.clip(y_pred, 0, 1))
    tp = K.sum(y_true * y_pred); fp = K.sum((1 - y_true) * y_pred); fn = K.sum(y_true * (1 - y_pred))
    precision = tp / (tp + fp + K.epsilon()); recall = tp / (tp + fn + K.epsilon())
    return 2 * (precision * recall) / (precision + recall + K.epsilon())

# Caminhos
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_PATH = os.path.abspath(os.path.join(BASE_PATH, '..'))
MODEL_DIR = os.path.join(PROJECT_PATH, 'model')
CLASSIFIER_PATH = os.path.join(MODEL_DIR, 'classifier_model', '00_classifier_model.h5')
OUT_DIR = os.path.join(MODEL_DIR, 'combined_model')
os.makedirs(OUT_DIR, exist_ok=True)

IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
ALPHA = 1.0


def rename_layers(model, prefix):
    for layer in model.layers:
        layer._name = prefix + layer.name
    return model


def build_combined():
    if not os.path.exists(CLASSIFIER_PATH):
        raise FileNotFoundError(
            f'Classificador final nao encontrado: {CLASSIFIER_PATH}\n'
            f'Rode antes: 01_train_final_model.py')

    print('==> Carregando MLP final')
    mlp = load_model(CLASSIFIER_PATH,
                     custom_objects={'f1_loss': get_f1_loss(), 'f1_metric': f1_metric},
                     compile=False)
    mlp = rename_layers(mlp, 'mlp_')

    print('==> Montando extrator MobileNetV1 (avg, congelado)')
    inp = Input(shape=IMAGE_SIZE + (IMAGE_CHANNELS,), name='input_image')
    base = MobileNet(weights='imagenet', include_top=False, pooling='avg',
                     input_shape=IMAGE_SIZE + (IMAGE_CHANNELS,), alpha=ALPHA)
    base = rename_layers(base, 'mobilenetv1_')
    base.trainable = False

    feats = base(inp)            # (None, 1024) -- avg pooling, SEM Flatten
    out = mlp(feats)             # (None, 1) -- P(obstaculo)
    combined = Model(inputs=inp, outputs=out, name='obstacle_detection_combined')
    combined.summary()

    h5 = os.path.join(OUT_DIR, 'classifier_model_combined.h5')
    combined.save(h5)
    print(f'    Modelo combinado salvo: {h5}')

    try:
        from tensorflow.keras.utils import plot_model
        plot_model(combined, to_file=os.path.join(OUT_DIR, 'model_combined_architecture.png'),
                   show_shapes=True)
    except Exception as e:
        print(f'    [aviso] plot_model ignorado: {e}')

    return combined


def export_tflite(combined):
    print('==> Convertendo para TFLite (float32)')
    converter = tf.lite.TFLiteConverter.from_keras_model(combined)
    # float32 padrao (sem quantizacao). Para INT8, ver 03_validate_tflite.py / comentarios.
    converter.optimizations = []  # explicitamente sem quantizacao
    tflite_model = converter.convert()
    path = os.path.join(OUT_DIR, 'classifier_model_combined.tflite')
    with open(path, 'wb') as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(path) / (1024 ** 2)
    print(f'    TFLite salvo: {path}  ({size_mb:.2f} MB)')
    print('\nOK. Proximo passo: 03_validate_tflite.py')


if __name__ == '__main__':
    model = build_combined()
    export_tflite(model)
