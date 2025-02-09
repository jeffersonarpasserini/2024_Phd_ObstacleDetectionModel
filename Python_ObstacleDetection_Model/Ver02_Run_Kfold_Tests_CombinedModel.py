import os
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import glob
import matplotlib

from tensorflow.keras import Input
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Normalization, LeakyReLU,Activation, ReLU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import BinaryCrossentropy


# Modelo de Extração - MobileNetV2
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score
from tensorflow.python.ops.metrics_impl import precision

from confusionMatrixCallback import ConfusionMatrixCallback
import combined_model
import extract_features
import metrics_view
from custom_binary_crossentropy import custom_binary_crossentropy

matplotlib.use('Agg')

os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

EXTENSAO_PERMITIDA = set(['png', 'jpg', 'jpeg'])

# Define o caminho base do projeto
BASE_PATH = os.path.dirname(os.path.abspath(__file__))

# Define os caminhos necessários
FEATURE_PATH = os.path.join(BASE_PATH, 'features')
MODEL_PATH = os.path.join(BASE_PATH, 'model', 'classifier_model')
COMBINED_MODEL_DIR = os.path.join(BASE_PATH, 'model', 'combined_model')
#TFLITE_MODEL_PATH = os.path.join(BASE_PATH, 'model', 'tflite_model')
RESULT_PATH_TRAINING = os.path.join(BASE_PATH, 'results_details', 'training_results')
RESULT_TEST_PATH = os.path.join(BASE_PATH, 'results_details', 'test_results')

DATASET_VIA_DATASET = os.path.join(BASE_PATH, 'C:\\Projetos\\2024_Phd_ObstacleDetectionModel\\via-dataset')
DATASET_VIA_DATASET_EXTENDED = os.path.join(BASE_PATH, 'C:\\Projetos\\2024_Phd_ObstacleDetectionModel\\via-dataset-extended')
DATASET_PATH = DATASET_VIA_DATASET


# Lista de diretórios para garantir que existem
paths_to_ensure = [
    FEATURE_PATH,    # Diretório de 'features'
    MODEL_PATH,     # Diretório de 'model'
    #TFLITE_MODEL_PATH,  # Diretório de 'tflite_combined_model'
    RESULT_PATH_TRAINING,  # Diretório de 'results_details/confusion_matrix_results'
    RESULT_TEST_PATH #Diretorio de test_results
]

# Garantir que todos os diretórios existam
for path in paths_to_ensure:
    os.makedirs(path, exist_ok=True)

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
def save_classifier_model(X_train, y_train, X_validation, y_validation, split_index=0):
    try:
        # remover todas as matrizes de confusao da pasta do resultado detalhado
        #remove_all_png_files(RESULT_PATH)

        normalization_layer = Normalization()
        normalization_layer.adapt(X_train)

        classifier = Sequential()
        classifier.add(Input(shape=(X_train.shape[1],)))
        classifier.add(Dense(128, kernel_regularizer=l2(0.005)))
        classifier.add(BatchNormalization())
        classifier.add(LeakyReLU(alpha=0.1))
        classifier.add(Dropout(0.2))

        classifier.add(Dense(64, kernel_regularizer=l2(0.001)))
        classifier.add(BatchNormalization())
        classifier.add(LeakyReLU(alpha=0.1))
        classifier.add(Dropout(0.2))

        classifier.add(Dense(32, kernel_regularizer=l2(0.0001)))
        classifier.add(BatchNormalization())
        classifier.add(ReLU())
        classifier.add(Dropout(0.2))

        classifier.add(Dense(16, kernel_regularizer=l2(0.005)))
        classifier.add(BatchNormalization())
        classifier.add(Activation('tanh'))
        classifier.add(Dropout(0.2))

        classifier.add(Dense(1, activation='sigmoid'))

        # Compilar o modelo
        # Configurando a função de perda com pos_weight
        #loss_fn = custom_binary_crossentropy(pos_weight=2.0)
        #classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'recall', 'precision'])
        #classifier.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy', 'recall', 'precision'])
        classifier.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy', 'recall', 'precision'])

        # Treinar o modelo com as features extraídas
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)

        # Adicionar o callback para matriz de confusão
        confusion_matrix_callback = ConfusionMatrixCallback(X_validation, y_validation, RESULT_PATH_TRAINING,
                                                            split_index)

        # Treinamento
        classifier.fit(
            X_train, y_train,
            epochs=1000,
            batch_size=64,
            validation_data=(X_validation, y_validation),
            callbacks=[early_stopping, lr_scheduler, confusion_matrix_callback]
        )

        print("Classifier created.")

        # Salvar o modelo Keras
        file_name_h5 = f"{split_index:02d}_classifier_model.h5"
        keras_model_path = os.path.join(MODEL_PATH, file_name_h5)
        classifier.save(keras_model_path)
        print("Classifier model saved h5 file.")

        # Converter o modelo Keras para TensorFlow Lite
        #converter = tf.lite.TFLiteConverter.from_keras_model(classifier)
        #tflite_model = converter.convert()

        # Salvar o modelo TensorFlow Lite
        #file_name_tflite = f"{split_index:02d}_classifier_model.tflite"
        #tflite_model_path_h5 = os.path.join(MODEL_PATH, file_name_tflite)
        #with open(tflite_model_path_h5, 'wb') as f:
        #    f.write(tflite_model)
        print("Classifier tflite model saved.")

    except Exception as e:
        return print(f"Erro: {str(e)}")


def save_combined_model(split_index=0):
    try:
        combined_model.generateCombinedModel(split_index)
        return print('Modelo combinado gerado e salvo com sucesso')
    except Exception as e:
        return print(f"Erro: {str(e)}")


def metrics():
    try:
        metrics_view.metricsView()
        return print('Métricas geradas')
    except Exception as e:
        return print(f"Erro: {str(e)}")


# Função para carregar o modelo e avaliar no conjunto de teste
def evaluate_model_on_test(X_test, y_test, split_index=0):
    """
        Avalia um modelo salvo em um conjunto de testes.

        Args:
            X_test (array-like): Conjunto de teste de características.
            y_test (array-like): Rótulos reais do conjunto de teste.
            split_index (int): Índice do modelo salvo para carregar.

        Returns:
            dict: Métricas calculadas.
    """
    try:
        # Verifica se o modelo existe
        file_name_h5 = f"{split_index:02d}_classifier_model.h5"
        DENSE_MODEL_PATH = os.path.join(MODEL_PATH, file_name_h5)
        if not os.path.exists(DENSE_MODEL_PATH):
            raise FileNotFoundError(f"Modelo não encontrado: {DENSE_MODEL_PATH}")

        # Carrega o modelo salvo
        model = load_model(DENSE_MODEL_PATH)
        print("Classifier model loaded.")

        # Realiza as predições no conjunto de teste
        y_pred_probs = model.predict(X_test)  # Probabilidades
        y_pred = (y_pred_probs > 0.5).astype(int)  # Converter para classes binárias

        # Métricas
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tn + fp + fn + tp)
        roc_auc = roc_auc_score(y_test, y_pred_probs)
        f1 = f1_score(y_test, y_pred)
        recall = tp / (tp + fp)
        precision = tp / (tp + fn)


        return {
            "Fold": split_index + 1,
            "Accuracy": accuracy,
            "ROC_AUC": roc_auc,
            "F1_Score": f1,
            "Recall": recall,
            "Precision": precision,
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        }

    except Exception as e:
        print(f"Erro ao avaliar o modelo: {e}")


def split_dataset(features, labels, kfold_data_train, kfold_data_test, validation_size=0.1, split_index=0):
    #separa os dados de teste das features (10% dos dados)
    X_test = np.array(features[kfold_data_test])
    y_test = np.array(labels[kfold_data_test])

    #separa dados de treinamento/validação das features (90% dos dados)
    X_temp = np.array(features[kfold_data_train])
    y_temp = np.array(labels[kfold_data_train])

    #Dividir os dados em treino e base validação (90% treino, 10% validação)
    X_train, X_validation, y_train, y_validation = train_test_split(X_temp, y_temp, test_size=validation_size, random_state=SEED)

    features_dict = {
        "X_train_features": X_train,
        "X_validation_features": X_validation,
        "X_test_features": X_test,
        "y_train_features": y_train,
        "y_validation_features": y_validation,
        "y_test_features": y_test,
    }

    save_features_to_csv(features_dict, FEATURE_PATH, split_index)

    return X_train, X_validation, X_test, y_train, y_validation, y_test

def save_features_to_csv(features_dict, base_path, split_index):
    """
    Salva arrays de features e rótulos em arquivos CSV.

    Args:
        features_dict (dict): Dicionário contendo os arrays a serem salvos.
                              As chaves devem ser os nomes dos arquivos.
        base_path (str): Caminho base onde os arquivos serão salvos.
        split_index (int): Índice do split atual para diferenciar os arquivos.

    Returns:
        None
    """
    for name, data in features_dict.items():
        file_path = os.path.join(base_path, f"{split_index:02d}_{name}.csv")
        pd.DataFrame(data).to_csv(file_path, index=False)
        print(f"Salvo: {file_path}")


def run():
    try:
        # Carregar os dados
        df = extract_features.load_data()
        # lables array
        labels = df["category"].replace({'clear': 1, 'non-clear': 0}).to_numpy().astype(int)
        print('Dataset carregado...')

        # creating folds for cross-validation - 10fold
        kfold_n_splits = 10
        kfold_n_repeats = 1
        kf = RepeatedKFold(n_splits=kfold_n_splits, n_repeats=kfold_n_repeats, random_state=SEED)
        kf.split(df)

        #limpa pasta de resultados de treinamento do classificador
        remove_all_png_files(MODEL_PATH)

        # dict para salvar resultados dos tests.
        results = []
        # arquivo para armazenar resultados dos testes.
        tests_results = os.path.join(RESULT_TEST_PATH, "tests_results.csv")

        # kfold loop
        for index, [train, test] in enumerate(kf.split(df)):
            # 10% de dados para testes
            # 90% para treinamento e validação

            print(f"Fold: {index}")

            features = extract_features.modular_extract_features(df)
            print('ModelMobileNetV2 loaded...')
            print('Features extracted...')

            #10% para teste, 10% para validação e 80% para treinamento
            X_train, X_validation, X_test, y_train, y_validation, y_test = split_dataset(features,labels,train,test,0.1, index)
            print('Dataset splitted...')

            print("X_train shape: ", X_train.shape)
            print("y_train shape: ", y_train.shape)
            print("X_val shape: ", X_validation.shape)
            print("y_val shape: ", y_validation.shape)
            print("X_test shape: ", X_test.shape)
            print("y_test shape: ", y_test.shape)

            #gera o modelo do classificador
            save_classifier_model(X_train,
                                  y_train,
                                  X_validation,
                                  y_validation, split_index=index)

            #gera o modelo combinado do classificador e extrator de caracteristicas
            #save_combined_model(split_index=index)

            #-------- Classification Dataset Tests ---------------#
            metrics = evaluate_model_on_test(X_test,
                                             y_test, split_index=index)

            if metrics:
                results.append(metrics)
                print(f"Fold: {metrics.get("Fold")} - Accuracy: {metrics.get("Accuracy")} - F1Score: {metrics.get("F1_Score")} - AUC: {metrics.get("ROC_AUC")} - Precision: {metrics.get("Precision")} - Recall: {metrics.get("Recall")}")

        # Salvar os resultados em um arquivo CSV
        results_df = pd.DataFrame(results)
        results_df.to_csv(tests_results, index=False)
        print(f"Resultados salvos em {tests_results}")

        return print('Classifier Process ended.')
    except Exception as e:
        return print(f"Erro: {str(e)}")


if __name__ == "__main__":
    run()