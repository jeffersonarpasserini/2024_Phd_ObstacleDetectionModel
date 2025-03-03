import datetime
import os
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
import random

from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, accuracy_score
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from scikeras.wrappers import KerasClassifier
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

# Garantir reprodutibilidade
os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Definir caminhos
BASE_PATH = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Dataset's
DATASET_VIA_DATASET = os.path.abspath(os.path.join(BASE_PATH, '..', 'via-dataset'))
DATASET_VIA_DATASET_EXTENDED = os.path.abspath(os.path.join(BASE_PATH, '..', 'via-dataset-extended'))

# Chaveamento entre os datasets
USE_EXTENDED_DATASET = True  # 🔹 Altere para False para usar o 'via-dataset'

DATASET_PATH = DATASET_VIA_DATASET_EXTENDED if USE_EXTENDED_DATASET else DATASET_VIA_DATASET

FEATURE_PATH = os.path.join(BASE_PATH, 'features')
RESULTS_PATH = os.path.join(BASE_PATH, 'results_details', 'loocv_results')
os.makedirs(RESULTS_PATH, exist_ok=True)

if not os.path.exists(DATASET_PATH):
    print(f"❌ ERRO: O caminho do dataset não existe: {DATASET_PATH}")
else:
    print(f"✅ Caminho do dataset encontrado: {DATASET_PATH}")


# Parâmetros globais - extrator de caracteristicas
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
POOLING = 'avg'
ALPHA = 1.0

# -------------- Extração de Caracteristicas -------------------------------
def load_data():
    # Definir extensões de arquivos válidos (imagens)
    valid_extensions = ('.jpg', '.jpeg', '.png')

    print(DATASET_PATH)

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

def get_extract_model(model_type):

    # Carrega o modelo e a função de pré-processamento
    if model_type == 'MobileNetV2':
        print('------------- Gera modelo MobileNetV2 ------------------')

        # Importando MobileNetV2
        from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input

        model = MobileNetV2(weights='imagenet', include_top=False, pooling=POOLING,
                            input_shape=IMAGE_SIZE + (IMAGE_CHANNELS,), alpha=ALPHA)

        preprocessing_function = preprocess_input

    elif model_type == 'MobileNetV1':
        print('------------- Gera modelo MobileNetV1 ------------------')

        # Importando MobileNetV1
        from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input

        model = MobileNet(
            weights='imagenet',
            include_top=False,  # Remove a camada de saída do ImageNet
            pooling=POOLING,  # Define o tipo de pooling na saída
            input_shape=IMAGE_SIZE + (IMAGE_CHANNELS,),
            alpha=ALPHA  # 🔹 Define a largura da rede (número de filtros convolucionais)
        )

        preprocessing_function = preprocess_input

    elif model_type == 'MobileNetV3Small':
        print('------------- Gera modelo MobileNetV3Small ------------------')
        # Importando MobileNetV3Small
        from tensorflow.keras.applications.mobilenet_v3 import MobileNetV3Small, preprocess_input

        model = MobileNetV3Small(
            weights='imagenet',
            include_top=False,  # Remove a camada de saída do ImageNet
            pooling=POOLING,  # Define o tipo de pooling na saída
            input_shape=IMAGE_SIZE + (IMAGE_CHANNELS,),
            alpha=ALPHA  # Controla o tamanho do modelo
        )

        preprocessing_function = preprocess_input

    elif model_type == 'MobileNetV3Large':
        print('------------- Gera modelo MobileNetV3Large ------------------')
        # Importando MobileNetV3Large
        from tensorflow.keras.applications.mobilenet_v3 import MobileNetV3Large, preprocess_input

        model = MobileNetV3Large(
            weights='imagenet',
            include_top=False,  # Remove a camada de saída do ImageNet
            pooling=POOLING,  # Define o tipo de pooling na saída
            input_shape=IMAGE_SIZE + (IMAGE_CHANNELS,),
            alpha=ALPHA  # Controla o tamanho do modelo
        )

        preprocessing_function = preprocess_input

    else:
        raise ValueError("Error: Model not implemented.")

    #output = Flatten()(model.layers[-1].output)
    #model = Model(inputs=model.inputs, outputs=output)

    return model, preprocessing_function

def extract_features(df, model, preprocessing_function, use_augmentation):
    """Extrai features das imagens do dataset, com Data Augmentation opcional."""

    df["category"] = df["category"].replace({1: 'clear', 0: 'non-clear'})

    if use_augmentation:
        print("🟢 Aplicando Data Augmentation...")
        datagen = ImageDataGenerator(
            preprocessing_function=preprocessing_function,
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode="nearest"
        )
    else:
        print("🔵 Sem Data Augmentation...")
        datagen = ImageDataGenerator(preprocessing_function=preprocessing_function)

    total = df.shape[0]
    batch_size = 4
    steps = int(np.ceil(total / batch_size))

    generator = datagen.flow_from_dataframe(
        df,
        DATASET_PATH,
        x_col='filename',
        y_col='category',
        class_mode='binary' if len(df['category'].unique()) == 2 else 'categorical',
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False
    )

    features = model.predict(generator, steps=steps)
    return features

def feature_model_extract(df, model_type, use_augmentation=False, use_shap=False, sample_size=None):
    """Executa o processo de extração de características, podendo incluir Data Augmentation e SHAP."""
    model, preprocessing_function = get_extract_model(model_type)

    features = extract_features(df, model, preprocessing_function, use_augmentation)  # ✅ Passar o parâmetro

    labels = df["category"].replace({'clear': 1, 'non-clear': 0}).to_numpy().astype(int)  # 🔹 Adicionar labels corretamente

    return features

# -------------------------------- CLASSIFICADOR  ---------------------------------------------------------
def get_classifier_model(input_shape, activation, dropout_rate, learning_rate, n_layers, n_neurons, optimizer):
    model = Sequential([Input(shape=(input_shape,))])
    for _ in range(n_layers):
        model.add(Dense(n_neurons, activation=activation))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    # Definir o otimizador com base no parâmetro
    if optimizer == 'adam':
        optimizer_instance = Adam(learning_rate=learning_rate)
    elif optimizer == 'rmsprop':
        optimizer_instance = RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("Optimizer não reconhecido. Use 'adam' ou 'rmsprop'.")

    model.compile(optimizer=optimizer_instance, loss=BinaryCrossentropy(), metrics=['accuracy'])
    return model

# Executa o teste Gridsearch
def run_GridSearch(extract_features_model, params_grid, lr_scheduler, features_test_size, features_validation_size, epochs):

    # Carregamento das imagens do dataset
    df = load_data()

    for model_type in (extract_features_model):
        print(f"Testing extract model ${model_type}...")

        labels = df["category"].replace({'clear': 1, 'non-clear': 0}).to_numpy().astype(int)
        features = feature_model_extract(df, model_type)

        # Dividir os dados em treino e teste
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=features_test_size, random_state=SEED)

        # Criar modelo para GridSearchCV incluindo callbacks corretamente
        model = KerasClassifier(
            model=lambda: get_classifier_model(
                input_shape=X_train.shape[1],
                activation='relu',
                dropout_rate=0.2,
                learning_rate=0.001,
                n_layers=2,
                n_neurons=128,
                optimizer='adam'
            ),
            verbose=0,
            callbacks=[lr_scheduler]
        )

        # Configurar GridSearchCV com os hiperparâmetros corrigidos
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=params_grid,
            scoring='accuracy',
            cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED),
            refit=True,  # Mantém o melhor modelo treinado
            n_jobs=1,
            verbose=0)

        # Executar a busca com validação cruzada
        grid_result = grid_search.fit(
            X_train, y_train,
            validation_split=features_validation_size,
            callbacks=[lr_scheduler]
        )

        # Criar DataFrame com os resultados do GridSearchCV
        results_df = pd.DataFrame(grid_result.cv_results_)
        results_df.to_csv(f"{model_type}_grid_search_results.csv", index=False)
        print(f"Resultados do GridSearchCV salvos em '{model_type}_grid_search_results.csv'")

        # Exibir os melhores hiperparâmetros encontrados
        print(f"{model_type} - Melhores hiperparâmetros: ", grid_search.best_params_)
        print(f"{model_type} - Melhor acurácia: ", grid_search.best_score_)

        # Avaliação de todos os modelos treinados com X_test
        test_results = []
        individual_classifications = []

        for idx, params in enumerate(grid_result.cv_results_['params']):
            print(f"🔍 Testando modelo {idx + 1} com parâmetros: {params}")

            # Remover o prefixo 'model__' dos parâmetros antes de passar para a função
            clean_params = {key.replace("model__", ""): value for key, value in params.items()}

            model = get_classifier_model(**clean_params)
            model.fit(X_train, y_train, epochs=epochs, callbacks=[lr_scheduler], verbose=0)

            y_pred = (model.predict(X_test) > 0.5).astype("int32")

            acc = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_pred)

            print(f"Extract Model: ${model_type} - Classif: ${idx + 1} - Parâmetros: ${clean_params} - Accuracy: ${acc}")

            test_results.append({
                "Modelo": idx + 1,
                "ExtractModel": model_type,
                "Parâmetros": clean_params,
                "Acurácia": acc,
                "Precisão": precision,
                "Recall": recall,
                "F1-Score": f1,
                "ROC-AUC": roc_auc
            })

            for i, pred in enumerate(y_pred):
                individual_classifications.append({
                    "Modelo": idx + 1,
                    "ExtractModel": model_type,
                    "Parâmetros": clean_params,
                    "Imagem": i,
                    #"Real": y_test.iloc[i],
                    "Real": y_test[i],
                    "Predito": pred[0]
                })

        # Salvar os resultados em arquivos CSV
        test_results_df = pd.DataFrame(test_results)
        test_results_df.to_csv(f"{model_type}_test_results_summary.csv", index=False)

        individual_classifications_df = pd.DataFrame(individual_classifications)
        individual_classifications_df.to_csv(f"{model_type}_test_results_individual.csv", index=False)

        print(f"📁 Resultados resumidos salvos em '{model_type}_test_results_summary.csv'")
        print(f"📁 Resultados individuais salvos em '{model_type}_test_results_individual.csv'")



if __name__ == "__main__":
    # ✅ Definir a lista de cnn extratoras de caracteristicas
    #extract_features_model = {"MobileNetV1", "MobileNetV2","MobileNetV3Small", "MobileNetV3Large"}
    extract_features_model = {"MobileNetV1"}

    # ✅ Definir a grade de hiperparâmetros
    #params_grid = {
    #   'model__learning_rate': [0.0005, 0.0001, 0.005, 0.001, 0.05, 0.01, 0.1],
    #    'model__activation': ['relu', 'tanh'],
    #    'model__n_layers': [1, 2, 3],
    #    'model__n_neurons': [16, 32, 64, 128, 256, 512],
    #    'model__dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    #    'model__optimizer': ['adam', 'rmsprop']
    #}

    params_grid = {
        'model__learning_rate': [0.0001],
        'model__activation': ['relu'],
        'model__n_layers': [1],
        'model__n_neurons': [128],
        'model__dropout_rate': [0.1],
        'model__optimizer': ['rmsprop']
    }

    # % test dataset
    features_test_size=0.2
    # % validation dataset
    features_validation_size=0.2
    # number trainning epochs in test dataset
    epochs=1000

    # configura o redutor na velocidade de convergencia do modelo.
    lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)

    # Para rodar teste de kfold cross validation - descomente abaixo
    run_GridSearch(extract_features_model, params_grid, lr_scheduler, features_test_size,
                   features_validation_size, epochs)
