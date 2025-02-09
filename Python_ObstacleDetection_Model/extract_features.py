import numpy as np
import pandas as pd
import tensorflow as tf
import random
import time
import os
from PIL import Image

from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Flatten


# Garantir reprodutibilidade dos resultados
os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Definindo paths
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

DATASET_VIA_DATASET = os.path.join(BASE_PATH, 'C:\\Projetos\\2024_Phd_ObstacleDetectionModel\\via-dataset')
DATASET_VIA_DATASET_EXTENDED = os.path.join(BASE_PATH, 'C:\\Projetos\\2024_Phd_ObstacleDetectionModel\\via-dataset-extended')
DATASET_PATH = DATASET_VIA_DATASET_EXTENDED
FEATURE_PATH = os.path.join(BASE_PATH, 'features')


def load_data():
    # Definir extensões de arquivos válidos (imagens)
    valid_extensions = ('.jpg', '.jpeg', '.png')

    # Listar apenas arquivos que possuem extensões de imagem válidas
    filenames = [f for f in os.listdir(DATASET_PATH) if f.lower().endswith(valid_extensions)]

    categories = []

    for filename in filenames:
        category = filename.split('.')[0]
        if category == 'clear':
            categories.append(1)
        else:
            categories.append(0)

    df = pd.DataFrame({
        'filename': filenames,
        'category': categories
    })

    return df

def extract_features(df, model, preprocessing_function, image_size):

    # Atualiza os nomes de categoria
    df["category"] = df["category"].replace({1: 'clear', 0: 'non-clear'})

    datagen = ImageDataGenerator(
        preprocessing_function=preprocessing_function
    )

    total = df.shape[0]
    batch_size = 4

    # Calcula o número correto de steps
    steps = int(np.ceil(total / batch_size))

    generator = datagen.flow_from_dataframe(
        df,
        DATASET_PATH,
        x_col='filename',
        y_col='category',
        class_mode='categorical',
        target_size=image_size,
        batch_size=batch_size,
        shuffle=False
    )

    # Realiza a predição com base no número de steps calculado
    features = model.predict(generator, steps=steps)

    return features

def create_model(model_type):

    IMAGE_CHANNELS = 3
    POOLING = 'avg'  # 'avg' pooling para MobileNetV2 --> None, avg, max
    image_size = (224, 224)
    alpha = 1.0

    # Carrega o modelo e a função de pré-processamento
    if model_type == 'MobileNetV2':
        print('------------- Gera modelo MobileNetV2 ------------------')

        from keras.api.applications.mobilenet_v2 import MobileNetV2, preprocess_input

        utiliza_GlobalAveragePooling2D = False


        if utiliza_GlobalAveragePooling2D:
            POOLING = "None"

            base_model = MobileNetV2(weights='imagenet', include_top=False, pooling=POOLING,
                                input_shape=image_size + (IMAGE_CHANNELS,), alpha=alpha)

            # Congelar as primeiras camadas do MobileNetV2
            #for layer in base_model.layers[:-10]:
            #    layer.trainable = False

            # Adiciona a camada GlobalAveragePooling2D à saída do base_model
            x = GlobalAveragePooling2D()(base_model.output)

            # Cria o modelo final, definindo as entradas e a saída após o pooling
            model = Model(inputs=base_model.input, outputs=x)

        else:
            model = MobileNetV2(weights='imagenet', include_top=False, pooling=POOLING,
                                     input_shape=image_size + (IMAGE_CHANNELS,), alpha=alpha)

            # Congelar as primeiras camadas do MobileNetV2
            #for layer in model.layers[:-10]:
            #    layer.trainable = False

        preprocessing_function = preprocess_input

    elif model_type == 'MobileNetV1':
        print('------------- Gera modelo MobileNetV1 ------------------')

        from keras.api.applications.mobilenet import MobileNet, preprocess_input

        model = MobileNet(
            weights='imagenet',
            include_top=False,  # Remove a camada de saída do ImageNet
            pooling=POOLING,  # Define o tipo de pooling na saída
            input_shape=image_size + (IMAGE_CHANNELS,),
            alpha=alpha  # 🔹 Define a largura da rede (número de filtros convolucionais)
        )

        preprocessing_function = preprocess_input

    elif model_type == 'MobileNetV3Small':
        print('------------- Gera modelo MobileNetV3Small ------------------')
        from tensorflow.keras.applications import MobileNetV3Small
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

        model = MobileNetV3Small(
            weights='imagenet',
            include_top=False,  # Remove a camada de saída do ImageNet
            pooling=POOLING,  # Define o tipo de pooling na saída
            input_shape=image_size + (IMAGE_CHANNELS,),
            alpha=alpha  # Controla o tamanho do modelo
        )

        preprocessing_function = preprocess_input

    elif model_type == 'MobileNetV3Large':
        print('------------- Gera modelo MobileNetV3Large ------------------')
        from tensorflow.keras.applications import MobileNetV3Large
        from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

        model = MobileNetV3Large(
            weights='imagenet',
            include_top=False,  # Remove a camada de saída do ImageNet
            pooling=POOLING,  # Define o tipo de pooling na saída
            input_shape=image_size + (IMAGE_CHANNELS,),
            alpha=alpha  # Controla o tamanho do modelo
        )

        preprocessing_function = preprocess_input

    else:
        raise ValueError("Error: Model not implemented.")



    #output = Flatten()(model.layers[-1].output)
    #model = Model(inputs=model.inputs, outputs=output)

    return model, preprocessing_function, image_size

def feature_model_extract(df, model_type='MobileNetV2'):
    start = time.time()

    # Extrai features usando MobileNetV2
    modelMobileNetV2, preprocessing_functionMobileNetV2, image_sizeMobileNetV2 = create_model(model_type)
    features_MobileNetV2 = extract_features(df, modelMobileNetV2, preprocessing_functionMobileNetV2, image_sizeMobileNetV2)

    end = time.time()

    time_feature_extraction = end - start

    return features_MobileNetV2, time_feature_extraction

#utilizado pelo modulo obstacleDetectionModel
def modular_extract_features(df, model_type='MobileNetV2'):
    # Extraindo as características das imagens
    features, time_feature_extraction = feature_model_extract(df, model_type)

    return features

# ----------------------- MAIN ------------------------------------------------
#utilizar para execução direta do programa extract_features.py
def main_extract_features():
    # Carregando as imagens em um dataframe
    df = load_data()

    # Extraindo as características das imagens
    features, time_feature_extraction = feature_model_extract(df)

    # Convertendo as características em um dataframe
    df_csv = pd.DataFrame(features)

    # Salvando o dataframe em um arquivo CSV
    df_csv.to_csv(FEATURE_PATH)

    print(f"Extração de features concluída em {time_feature_extraction:.2f} segundos.")

# Este bloco garante que o código seja executado apenas quando o arquivo for executado diretamente
if __name__ == "__main__":
    main_extract_features()