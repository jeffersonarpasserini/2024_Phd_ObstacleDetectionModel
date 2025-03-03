import datetime
import os
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
import random
import gc
import cv2

from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.model_selection import LeaveOneOut, train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

# Verifica se há GPUs disponíveis
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("✅ Memória da GPU configurada para alocação dinâmica.")
    except RuntimeError as e:
        print(f"🚨 Erro ao configurar a memória da GPU: {e}")
else:
    print("❌ Nenhuma GPU disponível. Rodando na CPU!")

# Garantir reprodutibilidade
SEED = 1980
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Definir caminhos
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, '..', 'via-dataset-extended')
RESULTS_PATH = os.path.join(BASE_PATH, 'results_details', 'gradcam_results')
os.makedirs(RESULTS_PATH, exist_ok=True)

# Parâmetros globais
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
POOLING = 'avg'
ALPHA = 1.0


def load_data():
    valid_extensions = ('.jpg', '.jpeg', '.png')
    filenames = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith(valid_extensions)]
    categories = [1 if "clear" in f else 0 for f in filenames]
    df = pd.DataFrame({'filename': filenames, 'category': categories})
    return df


def get_extract_model(model_type):
    with tf.device('/GPU:0'):
        if model_type == 'MobileNetV1':
            from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
            model = MobileNet(weights='imagenet', include_top=True, input_shape=IMAGE_SIZE + (IMAGE_CHANNELS,),
                              alpha=ALPHA)
            preprocessing_function = preprocess_input
        elif model_type == 'MobileNetV2':
            from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
            model = MobileNetV2(weights='imagenet', include_top=True, input_shape=IMAGE_SIZE + (IMAGE_CHANNELS,),
                                alpha=ALPHA)
            preprocessing_function = preprocess_input
        else:
            raise ValueError("Modelo não implementado.")

    return model, preprocessing_function


def compute_gradcam(model, img_array, layer_name="conv_pw_13_relu"):
    """Computa o mapa Grad-CAM para uma imagem de entrada."""
    grad_model = Model(inputs=model.inputs, outputs=[model.get_layer(layer_name).output, model.output])

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = np.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)[0]

    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    return heatmap, class_idx  # 🔹 Removido `.numpy()`


import cv2

def overlay_gradcam(img_path, heatmap, alpha=0.4):
    """Sobrepõe o mapa de ativação Grad-CAM na imagem original."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Corrige a ordem de cores para RGB

    # 🔹 Redimensiona o heatmap para o tamanho da imagem original
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # 🔹 Normaliza o heatmap para o intervalo 0-255
    heatmap = np.uint8(255 * heatmap)

    # 🔹 Converte o heatmap para um mapa de cores RGB
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    # 🔹 Combina a imagem original com o heatmap
    overlay = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

    return overlay



def analyze_gradcam(model, df, preprocessing_function, sample_size=10):
    """Executa a análise Grad-CAM para explicar previsões do modelo."""
    print(f"🔍 Executando Grad-CAM para {sample_size} imagens.")

    sample_df = df.sample(n=sample_size, random_state=SEED)
    image_paths = [os.path.join(DATASET_PATH, fname) for fname in sample_df["filename"]]

    for img_path in image_paths:
        img = image.load_img(img_path, target_size=IMAGE_SIZE)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocessing_function(img_array)

        heatmap, class_idx = compute_gradcam(model, img_array)

        overlay = overlay_gradcam(img_path, heatmap)

        # 📊 Salva imagem com Grad-CAM
        plt.figure(figsize=(6, 6))
        plt.imshow(overlay)
        plt.axis('off')
        plt.title(f"Grad-CAM - Classe {class_idx}")
        save_path = os.path.join(RESULTS_PATH, f"gradcam_{os.path.basename(img_path)}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    print(f"📊 Mapas Grad-CAM salvos em: {RESULTS_PATH}")


def feature_model_extract(df, model_type, analyze_model=False, sample_size=None):
    """Executa o processo de extração de características e análise do modelo."""
    model, preprocessing_function = get_extract_model(model_type)

    if analyze_model:
        analyze_gradcam(model, df, preprocessing_function, sample_size)

    return model


if __name__ == "__main__":
    model_type = 'MobileNetV1'
    analyze_model = True
    sample_size = 1000

    df = load_data()
    feature_model_extract(df, model_type, analyze_model, sample_size)
