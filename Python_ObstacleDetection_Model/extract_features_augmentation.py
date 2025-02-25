import os
import time
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import shap
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K

# Garantir reprodutibilidade
os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Definir caminhos
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_PATH, 'C:\\Projetos\\2024_Phd_ObstacleDetectionModel\\via-dataset-extended')
FEATURE_PATH = os.path.join(BASE_PATH, 'features')

# Parâmetros globais
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
POOLING = 'avg'
ALPHA = 1.0  # Parâmetro para MobileNet


def load_data():
    """Carrega o dataset e define categorias com base no nome do arquivo."""
    valid_extensions = ('.jpg', '.jpeg', '.png')
    filenames = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith(valid_extensions)]
    categories = [1 if filename.startswith('clear') else 0 for filename in filenames]
    return pd.DataFrame({'filename': filenames, 'category': categories})


def get_model(model_type):
    """Carrega o modelo de extração de features de acordo com o tipo selecionado."""
    model_map = {
        'MobileNetV1': ('mobilenet', 'MobileNet'),
        'MobileNetV2': ('mobilenet_v2', 'MobileNetV2'),
        'MobileNetV3Small': ('mobilenet_v3', 'MobileNetV3Small'),
        'MobileNetV3Large': ('mobilenet_v3', 'MobileNetV3Large')
    }

    if model_type not in model_map:
        raise ValueError("Modelo não implementado.")

    module_name, class_name = model_map[model_type]
    model_module = __import__(f'tensorflow.keras.applications.{module_name}', fromlist=[class_name])
    preprocess_module = __import__(f'tensorflow.keras.applications.{module_name}', fromlist=['preprocess_input'])

    base_model = getattr(model_module, class_name)(
        weights='imagenet', include_top=False, pooling=POOLING,
        input_shape=IMAGE_SIZE + (IMAGE_CHANNELS,), alpha=ALPHA
    )

    return base_model, getattr(preprocess_module, 'preprocess_input')


def analyze_shap(model, features, labels, sample_size=None):
    """Executa análise SHAP para visualizar importância das características nas predições do classificador."""
    if sample_size is None:
        sample_size = int(0.05 * features.shape[0])  # 5% do total
    sample_size = max(50, min(sample_size, 500))  # Garante que está dentro dos limites

    print(f"🔍 Executando SHAP com {sample_size} imagens.")
    sample_idx = np.random.choice(features.shape[0], sample_size, replace=False)
    sample_features = features[sample_idx]
    sample_labels = labels[sample_idx]

    explainer = shap.Explainer(model, sample_features)
    shap_values = explainer(sample_features)

    shap.summary_plot(shap_values, sample_features)
    save_path = os.path.join(BASE_PATH, "shap_summary_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"📊 Gráfico SHAP salvo em: {save_path}")
    plt.close()


def feature_model_extract(df, model_type, use_augmentation=False, use_shap=False, sample_size=None):
    """Executa o processo de extração de características."""
    model, preprocessing_function = get_model(model_type)
    features = extract_features(df, model, preprocessing_function)
    labels = df['category'].to_numpy()

    if use_shap:
        analyze_shap(model, features, labels, sample_size)

    return features


def modular_extract_features(df, model_type, use_augmentation=False, use_shap=False, sample_size=None):
    """Interface modular para extração de características."""
    return feature_model_extract(df, model_type, use_augmentation, use_shap, sample_size)


def main_extract_features(model_type, use_augmentation=False, use_shap=False, sample_size=None):
    """Fluxo principal para extração e salvamento de características."""
    df = load_data()
    features = feature_model_extract(df, model_type, use_augmentation, use_shap, sample_size)
    pd.DataFrame(features).to_csv(FEATURE_PATH, index=False)
    print("Extração concluída.")


if __name__ == "__main__":
    main_extract_features('MobileNetV1', use_augmentation=False, use_shap=True, sample_size=100)
