import os
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import glob
import matplotlib

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split

from confusionMatrixCallback import ConfusionMatrixCallback
import combined_model
import extract_features
import metrics_view

matplotlib.use('Agg')

os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

EXTENSAO_PERMITIDA = set(['png', 'jpg', 'jpeg'])


import os

# Define o caminho base do projeto
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Define os caminhos necessários
FEATURE_PATH = os.path.join(BASE_PATH, 'features', 'features.csv')
MODEL_PATH = os.path.join(BASE_PATH, 'model', 'classifier_model')
TFLITE_MODEL_PATH = os.path.join(BASE_PATH, 'model', 'tflite_model')
RESULT_PATH = os.path.join(BASE_PATH, 'results_details', 'confusion_matrix_results')

# Lista de diretórios para garantir que existem
paths_to_ensure = [
    os.path.dirname(FEATURE_PATH),    # Diretório de 'features'
    MODEL_PATH,     # Diretório de 'model'
    TFLITE_MODEL_PATH,  # Diretório de 'tflite_combined_model'
    RESULT_PATH                      # Diretório de 'results_details/confusion_matrix_results'
]

# Garantir que todos os diretórios existam
for path in paths_to_ensure:
    os.makedirs(path, exist_ok=True)

data_filename = RESULT_PATH + "data_detailed.csv"

image_size = (224, 224)

def remove_all_png_files(directory):
    png_files = glob.glob(os.path.join(directory, "*.png"))
    for file_path in png_files:
        try:
            os.remove(file_path)
            print(f"Removido: {file_path}")
        except Exception as e:
            print(f"Erro ao remover {file_path}: {e}")

def load_features():
    # Carregar o CSV sem a restrição de colunas
    df = pd.read_csv(FEATURE_PATH, sep=',')

    # Contar o número de colunas
    num_cols = df.shape[1]
    #print(f"O número de colunas no arquivo CSV é: {num_cols}")

    # Carregar novamente o CSV, agora com o número correto de colunas
    df = pd.read_csv(FEATURE_PATH, sep=',', usecols=range(1, num_cols))

    print('Features carregadas com sucesso')

    return df

# Callback personalizado para gerar matriz de confusão ao final de cada epoch
def save_classifier_model(X_train, y_train, X_validation, y_validation):
    try:
        # remover todas as matrizes de confusao da pasta do resultado detalhado
        remove_all_png_files(RESULT_PATH)

        # Criar o modelo com base nas features carregadas
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1],)))  # Define explicitamente a entrada
        model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='sigmoid'))  # Saída binária

        # Compilar o modelo
        #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

        # Treinar o modelo com as features extraídas
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        # Adicionar o callback para matriz de confusão
        confusion_matrix_callback = ConfusionMatrixCallback(X_validation, y_validation, RESULT_PATH)

        model.fit(
            X_train,
            y_train,
            epochs=35,
            batch_size=64,
            validation_data=(X_validation, y_validation),
            callbacks=[early_stopping, confusion_matrix_callback]
        )

        # Salvar o modelo Keras
        keras_model_path = os.path.join(MODEL_PATH, 'classifier_model.h5')
        model.save(keras_model_path)

        # Converter o modelo Keras para TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()

        # Salvar o modelo TensorFlow Lite
        tflite_model_path_h5 = os.path.join(TFLITE_MODEL_PATH, 'classifier_model.tflite')
        with open(tflite_model_path_h5, 'wb') as f:
            f.write(tflite_model)

        return print(f"Modelo treinado e salvo com sucesso. Caminho do modelo TFLite: {TFLITE_MODEL_PATH}")

    except Exception as e:
        return print(f"Erro: {str(e)}")


def save_combined_model():
    try:
        combined_model.generateCombinedModel()
        return print('Modelo combinado gerado e salvo com sucesso')
    except Exception as e:
        return print(f"Erro: {str(e)}")


def metrics():
    try:
        metrics_view.metricsView()
        return print('Métricas geradas')
    except Exception as e:
        return print(f"Erro: {str(e)}")


def run():
    try:
        extract_features.main_extract_features()
        print('Features extraídas e salvas com sucesso')

        # Carregar os dados
        df = extract_features.load_data()
        print('Features carregadas com sucesso')

        # Convertendo os valores da coluna 'category' para strings adequadas para 'binary' mode
        df['category'] = df['category'].map({1: 'clear', 0: 'non-clear'})
        #df['category'] = df['category'].replace({1: 'clear', 0: 'non-clear'})  # Converte para strings apenas para ImageDataGenerator

        # Carregar as features já extraídas do arquivo CSV
        features_df = load_features()
        features = features_df.to_numpy()

        # converta os rótulos de volta para 0 e 1
        df['category'] = df['category'].map({'clear': 1, 'non-clear': 0})

        # Dividir os dados em treino e base temporaria (80% treino, 20% validação+teste)
        X_train, X_temp, y_train, y_temp = train_test_split(features, df['category'], test_size=0.2, random_state=SEED)

        # Dividir dados temporarios em validação (10%) e teste (10%)
        X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)

        print("X_train shape: ", X_train.shape)
        print("y_train shape: ", y_train.shape)
        print("X_val shape: ", X_val.shape)
        print("y_val shape: ", y_val.shape)
        print("X_test shape: ", X_test.shape)
        print("y_test shape: ", y_test.shape)

        save_classifier_model(X_train, y_train, X_val, y_val)

        save_combined_model()

        return print('Features extraídas, modelo gerado')
    except Exception as e:
        return print(f"Erro: {str(e)}")


if __name__ == "__main__":
    run()