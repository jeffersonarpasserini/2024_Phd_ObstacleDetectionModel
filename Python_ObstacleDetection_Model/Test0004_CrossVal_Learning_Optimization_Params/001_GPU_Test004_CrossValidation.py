import datetime
import os
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
import random
import time
import gc
import psutil  # para monitorar uso da RAM
import subprocess  # para chamar o nvidia-smi (GPU)

from itertools import product
from datetime import datetime


from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.model_selection import LeaveOneOut, GridSearchCV, ParameterGrid
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
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from scikeras.wrappers import KerasClassifier
from tqdm import tqdm
from scipy.stats import iqr

# faz o tf rodar na CPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configurar alocação dinâmica da memória na GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            #tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8192)])  # Ajuste para o seu hardware
        print("✅ Alocação dinâmica de memória ativada para a GPU.")
    except RuntimeError as e:
        print(f"❌ Erro ao configurar memória dinâmica da GPU: {e}")


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
RESULTS_PATH = os.path.join(BASE_PATH, 'results_details', 'gridsearch_results')
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
    print("extracting model..."+model_type+" Polling: "+POOLING)
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

    if POOLING == 'None':
        x = Flatten()(model.output)
        model = Model(inputs=model.input, outputs=x)

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

# -------------------------- Gerenciamento de Memória -----------------------------------------------------
def print_memory_usage():
    # Memória RAM
    ram = psutil.virtual_memory()
    ram_used = (ram.total - ram.available) / (1024**3)  # em GB
    print(f"🧠 RAM usada: {ram_used:.2f} GB")

    # Memória da GPU (NVIDIA)
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode == 0:
            gpu_memory = int(result.stdout.strip())
            print(f"🎮 GPU usada: {gpu_memory} MB")
        else:
            print("⚠️ GPU monitoramento não disponível (nvidia-smi não encontrado ou erro).")
    except FileNotFoundError:
        print("⚠️ Comando 'nvidia-smi' não encontrado. Sem monitoramento de GPU.")


# -------------------------------- CLASSIFICADOR  ---------------------------------------------------------
def get_classifier_model(input_shape, activation='relu', dropout_rate=0.1, learning_rate=0.001,
                         n_layers=1, n_neurons=128, optimizer='adam'):
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

def run_CrossValidation(extract_features_model, param_list, splits, features_validation_size, optimization_grid):
    print("\n=========================== CROSS VALIDATION (10 folds) ===========================")
    df = load_data()

    for model_type in extract_features_model:
        print(f"\n🔍 Testando modelo extrator: {model_type}...")
        start_model_type = time.time()

        labels = df["category"].replace({'clear': 1, 'non-clear': 0}).to_numpy().astype(int)
        features = feature_model_extract(df, model_type)

        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=SEED)

        fold = 1
        total_models = len(param_list) * len(optimization_grid) * splits
        print(f"📊 Total de execuções previstas: {total_models}")

        model_count = 1
        elapsed_time_total = 0

        all_results = []
        for train_index, test_index in skf.split(features, labels):
            X_train, X_test = features[train_index], features[test_index]
            y_train, y_test = labels[train_index], labels[test_index]

            for idx, params in tqdm(enumerate(param_list), total=len(list(param_list))):
                clean_params = {key.replace("model__", ""): value for key, value in params.items()}


                for early_pat, reduce_factor, reduce_pat, batch_size, epochs in optimization_grid:
                    print("***************************************************************************************************************************")
                    print(f"[INFO] Testando modelo {model_count}/{total_models} - Fold:{fold}")
                    print(f"[INFO] Model - {model_type} - Pooling: {POOLING}- Parâmetros: {clean_params}")
                    print("---------------------------------------------------------------------------------------------------------------------------")
                    print(f"[INFO] Learning - Parâmetros: Early Stopping - Patiente:{early_pat}")
                    print(f"[INFO] Learning - Parâmetros: Reduce on Plateau - Factor:{reduce_factor} - Patiente:{reduce_pat}")
                    print(f"[INFO] Model Fit - Parâmetros: Batch Size:{batch_size} - Epochs:{epochs}")
                    try:
                        model = get_classifier_model(input_shape=X_train.shape[1], **clean_params)

                        callbacks = [
                            EarlyStopping(monitor='val_loss', patience=early_pat, restore_best_weights=True),
                            ReduceLROnPlateau(monitor='val_loss', factor=reduce_factor, patience=reduce_pat, min_lr=1e-5)
                        ]

                        start_time = time.time()

                        model.fit(X_train, y_train,
                                  validation_split=features_validation_size,
                                  epochs=epochs,
                                  batch_size=batch_size,
                                  callbacks=callbacks,
                                  verbose=0)

                        y_pred = (model.predict(X_test) > 0.5).astype("int32")

                        elapsed = time.time() - start_time
                        elapsed_time_total += elapsed

                        remaining_time = (elapsed_time_total/model_count)*(total_models - model_count)
                        remaining = (total_models - model_count) * elapsed
                        print(f"[INFO] Tempo de processamento: {elapsed / 60:.2f} minutos")
                        print(f"[INFO] Tempo estimado restante pelo ultimo: {remaining / 60:.2f} minutos")
                        print(f"[INFO] Tempo estimado restante pelo média: {remaining_time / 60:.2f} minutos")
                        print("---------------------------------------------------------------------------------------------------------------------------")
                        all_results.append({
                            "Fold": fold,
                            "ExtractModel": model_type,
                            "Pooling": POOLING,
                            "Model_Parameters": clean_params,
                            "Epochs": epochs,
                            "Batch_Size": batch_size,
                            "EarlyStop_Patience": early_pat,
                            "ReduceLR_Factor": reduce_factor,
                            "ReduceLR_Patience": reduce_pat,
                            "Accuracy": accuracy_score(y_test, y_pred),
                            "Precision": precision_score(y_test, y_pred),
                            "Recall": recall_score(y_test, y_pred),
                            "F1-Score": f1_score(y_test, y_pred),
                            "ROC-AUC": roc_auc_score(y_test, y_pred)
                        })

                        #limpeza de memória
                        print("Limpando a memória ...")
                        del model, y_pred, callbacks, clean_params, early_pat, reduce_factor, reduce_pat, batch_size, epochs
                        K.clear_session()
                        gc.collect()

                        # memoria apos a limpeza
                        print_memory_usage()

                    except Exception as e:
                        print(f"[ERROR] Falha ao treinar/testar modelo {model_count}: {e}")
                        try:
                            del model, y_pred, callbacks
                        except:
                            pass
                        # limpeza de memória
                        K.clear_session()
                        gc.collect()

                    model_count += 1

                    # Salvar a cada 360 modelos testados
                    if model_count % 360 == 0:
                        temp_filename = f"temp_results_{model_count}.csv"
                        pd.DataFrame(all_results).to_csv(temp_filename, index=False)
                        print(f"💾 Resultados salvos em {temp_filename} (temporário)")
                        all_results.clear()
                        gc.collect()

            fold += 1

        elapsed_model_type = time.time() - start_model_type
        print(f"⏱ Tempo total para {model_type}: {elapsed_model_type / 60:.2f} minutos")

def merge_partial_results(path="./", output_name="CrossVal_CombinedResults.csv"):
    import glob
    partial_files = sorted(glob.glob(os.path.join(path, "temp_results_*.csv")))

    if not partial_files:
        print("⚠️ Nenhum arquivo de resultado temporário encontrado.")
        return

    dfs = [pd.read_csv(f) for f in partial_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_name, index=False)
    print(f"✅ Arquivo final salvo como: {output_name}")

if __name__ == "__main__":

    # ✅ Definir a lista de CNNs extratoras de características
    extract_features_model = {"MobileNetV1"}

    #modelos selecionados no Test0003
    custom_model_params = [
    # {'model__activation': 'relu', 'model__dropout_rate': 0.0, 'model__learning_rate': 0.001, 'model__n_layers': 1, 'model__n_neurons': 128, 'model__optimizer': 'adam'},
    # {'model__activation': 'relu', 'model__dropout_rate': 0.0, 'model__learning_rate': 0.05, 'model__n_layers': 1, 'model__n_neurons': 128, 'model__optimizer': 'rmsprop'},
    # {'model__activation': 'relu', 'model__dropout_rate': 0.1, 'model__learning_rate': 0.0005, 'model__n_layers': 1, 'model__n_neurons': 512, 'model__optimizer': 'adam'},
    # {'model__activation': 'relu', 'model__dropout_rate': 0.1, 'model__learning_rate': 0.0005, 'model__n_layers': 2, 'model__n_neurons': 128, 'model__optimizer': 'adam'},
    # {'model__activation': 'relu', 'model__dropout_rate': 0.1, 'model__learning_rate': 0.0005, 'model__n_layers': 2, 'model__n_neurons': 256, 'model__optimizer': 'adam'},
    # {'model__activation': 'relu', 'model__dropout_rate': 0.1, 'model__learning_rate': 0.001, 'model__n_layers': 1, 'model__n_neurons': 256, 'model__optimizer': 'adam'}.

     {'model__activation': 'relu', 'model__dropout_rate': 0.2, 'model__learning_rate': 0.0001, 'model__n_layers': 1, 'model__n_neurons': 128, 'model__optimizer': 'adam'}

    # {'model__activation': 'relu', 'model__dropout_rate': 0.2, 'model__learning_rate': 0.0001, 'model__n_layers': 1, 'model__n_neurons': 128, 'model__optimizer': 'rmsprop'},
    # {'model__activation': 'relu', 'model__dropout_rate': 0.2, 'model__learning_rate': 0.0001, 'model__n_layers': 1, 'model__n_neurons': 512, 'model__optimizer': 'adam'},
    # {'model__activation': 'relu', 'model__dropout_rate': 0.2, 'model__learning_rate': 0.0005, 'model__n_layers': 2, 'model__n_neurons': 512, 'model__optimizer': 'adam'},
    # {'model__activation': 'relu', 'model__dropout_rate': 0.2, 'model__learning_rate': 0.001, 'model__n_layers': 3, 'model__n_neurons': 512, 'model__optimizer': 'adam'},
    # {'model__activation': 'relu', 'model__dropout_rate': 0.3, 'model__learning_rate': 0.0001, 'model__n_layers': 1, 'model__n_neurons': 256, 'model__optimizer': 'adam'},
    # {'model__activation': 'relu', 'model__dropout_rate': 0.3, 'model__learning_rate': 0.0005, 'model__n_layers': 1, 'model__n_neurons': 128, 'model__optimizer': 'adam'},
    # {'model__activation': 'relu', 'model__dropout_rate': 0.3, 'model__learning_rate': 0.001, 'model__n_layers': 1, 'model__n_neurons': 256, 'model__optimizer': 'adam'},
    # {'model__activation': 'relu', 'model__dropout_rate': 0.3, 'model__learning_rate': 0.001, 'model__n_layers': 1, 'model__n_neurons': 512, 'model__optimizer': 'adam'},
    # {'model__activation': 'relu', 'model__dropout_rate': 0.3, 'model__learning_rate': 0.001, 'model__n_layers': 3, 'model__n_neurons': 128, 'model__optimizer': 'adam'},
    # {'model__activation': 'relu', 'model__dropout_rate': 0.3, 'model__learning_rate': 0.005, 'model__n_layers': 1, 'model__n_neurons': 256, 'model__optimizer': 'adam'},
    # {'model__activation': 'relu', 'model__dropout_rate': 0.4, 'model__learning_rate': 0.0001, 'model__n_layers': 1, 'model__n_neurons': 512, 'model__optimizer': 'adam'},
    # {'model__activation': 'relu', 'model__dropout_rate': 0.4, 'model__learning_rate': 0.0001, 'model__n_layers': 3, 'model__n_neurons': 128, 'model__optimizer': 'adam'},
    # {'model__activation': 'relu', 'model__dropout_rate': 0.4, 'model__learning_rate': 0.001, 'model__n_layers': 1, 'model__n_neurons': 256, 'model__optimizer': 'adam'},
    # {'model__activation': 'relu', 'model__dropout_rate': 0.4, 'model__learning_rate': 0.001, 'model__n_layers': 2, 'model__n_neurons': 128, 'model__optimizer': 'adam'},
    # {'model__activation': 'relu', 'model__dropout_rate': 0.5, 'model__learning_rate': 0.001, 'model__n_layers': 1, 'model__n_neurons': 256, 'model__optimizer': 'adam'}
    ]

    # custom_model_params = [
    #     {'model__activation': 'relu', 'model__dropout_rate': 0.0, 'model__learning_rate': 0.001, 'model__n_layers': 1,
    #      'model__n_neurons': 128, 'model__optimizer': 'adam'}]

    # Parâmetros gerais
    features_test_size = 0.2
    features_validation_size = 0.2
    splits = 10

    # Parametros de Otimização do treinamento
    optimization_grid = list(product(
        [50, 100, 150, 200],  # earlyStop_patience
        [0.5, 0.3],  # reduceLR_factor
        [5, 10, 15],  # reduceLR_patience
        [16, 32, 64],  # batch_size
        [100, 250, 500, 1000, 1500]  # epochs
    ))

    # optimization_grid = list(product(
    #     [150],  # earlyStop_patience
    #     [0.5],  # reduceLR_factor
    #     [10],  # reduceLR_patience
    #     [64],  # batch_size
    #     [1500]  # epochs
    # ))

    print(f"⚙️ Executando CrossValidation com {splits} splits...")
    run_CrossValidation(
        extract_features_model,
        custom_model_params,
        splits,
        features_validation_size,
        optimization_grid
    )

    #partial files - Combined Results
    merge_partial_results()





