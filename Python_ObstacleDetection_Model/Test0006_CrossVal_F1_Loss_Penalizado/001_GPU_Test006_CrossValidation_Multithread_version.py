import datetime
import math
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
import multiprocessing

from itertools import product

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

FEATURE_PATH = os.path.join(BASE_PATH, 'Test0005_CrossVal_F1_Loss', 'features')
RESULTS_PATH = os.path.join(BASE_PATH, 'Test0005_CrossVal_F1_Loss', 'results_details')

if not os.path.exists(DATASET_PATH):
    print(f"❌ ERRO: O caminho do dataset não existe: {DATASET_PATH}")
else:
    print(f"✅ Caminho do dataset encontrado: {DATASET_PATH}")

# 🔐 Cria diretórios comuns fora dos subprocessos
os.makedirs(FEATURE_PATH, exist_ok=True)
os.makedirs(RESULTS_PATH, exist_ok=True)

# Parâmetros globais - extrator de caracteristicas
IMAGE_SIZE = (224, 224)
IMAGE_CHANNELS = 3
POOLING = 'avg'
ALPHA = 1.0

# ----------------- Configurar alocação dinâmica da memória na GPU -------------------------------
def configurar_gpu_para_processos(paralelos=2, reserva_mb=4096):
    """Configura a GPU com limite de memória proporcional para execução paralela."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if not gpus:
        print("⚠️ Nenhuma GPU encontrada.")
        return

    try:
        # Consulta a memória total via nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        total_memory_mb = int(result.stdout.strip().split("\n")[0])  # em MB
        memoria_por_processo = max((total_memory_mb - reserva_mb) // paralelos, 1024)

        print(f"🧠 Memória total da GPU: {total_memory_mb} MB")
        print(f"🔒 Reservando {reserva_mb} MB para o sistema")
        print(f"🚀 Alocando {memoria_por_processo} MB para cada processo paralelo ({paralelos} processos)")

        for gpu in gpus:
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memoria_por_processo)]
            )
        print("✅ Configuração dinâmica da GPU aplicada.")
    except Exception as e:
        print(f"❌ Erro ao configurar GPU: {e}")


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

# ---------------- Calulo da função de perda no processo de aprendizado ------------------------------
def f1_metric(y_true, y_pred):
    """Calcula o F1-Score como métrica binária"""
    y_true = K.cast(y_true, 'float32')
    y_pred = K.round(K.clip(y_pred, 0, 1))

    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

    return f1


def get_f1_loss_com_penalizacao_fn(peso_fn=2.0, alpha=0.6, beta=1.0):
    def f1_loss(y_true, y_pred):
        y_true = K.cast(y_true, 'float32')
        y_pred = K.clip(y_pred, 0, 1)

        tp = K.sum(y_true * y_pred)
        fp = K.sum((1 - y_true) * y_pred)
        fn = K.sum(y_true * (1 - y_pred))

        # Penalização aplicada diretamente aos FNs
        fn_ponderado = peso_fn * fn

        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn_ponderado + K.epsilon())

        f1 = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + K.epsilon())

        return 1 - (alpha * f1 + (1 - alpha) * precision)

    return f1_loss

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
            categories.append(0)
        else:
            categories.append(1)

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

    df["category"] = df["category"].replace({0: 'clear', 1: 'non-clear'})

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
    """Executa extração de características com cache baseado no modelo."""
    cache_name = f"features_{model_type}_{POOLING}.npz"
    cache_path = os.path.join(FEATURE_PATH, cache_name)
    os.makedirs(FEATURE_PATH, exist_ok=True)

    if os.path.exists(cache_path):
        print(f"📦 Carregando features do cache: {cache_path}")
        data = np.load(cache_path)
        features = data["features"]
        labels = data["labels"]
    else:
        print(f"🚀 Extraindo features para: {model_type} (sem cache)")
        model, preprocessing_function = get_extract_model(model_type)
        features = extract_features(df, model, preprocessing_function, use_augmentation)
        labels = df["category"].replace({'clear': 0, 'non-clear': 1}).to_numpy().astype(int)

        np.savez_compressed(cache_path, features=features, labels=labels)
        print(f"💾 Features salvos em: {cache_path}")

    return features, labels

# -------------------------------- CLASSIFICADOR  ---------------------------------------------------------
def get_classifier_model(input_shape, activation='relu', dropout_rate=0.1, learning_rate=0.001,
                         n_layers=1, n_neurons=128, optimizer='adam', f1_loss_used=False, f1_alpha=0.6, f1_beta=1.0, peso_penalty_fn=2.0):

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

    if (f1_loss_used):
        loss_fn = get_f1_loss_com_penalizacao_fn(peso_fn=peso_penalty_fn, alpha=f1_alpha, beta=f1_beta)
        model.compile(optimizer=optimizer_instance, loss=loss_fn, metrics=['accuracy', f1_metric])
    else:
        model.compile(optimizer=optimizer_instance, loss=BinaryCrossentropy(), metrics=['accuracy'])

    return model

def run_CrossValidation(custom_model_params, splits, features_validation_size, start=0, end=None):
    print("\n=========================== CROSS VALIDATION (10 folds) ===========================")
    df = load_data()

    elapsed_time_total = 0
    results = []

    # Slice do conjunto de modelos a testar
    modelos_testar = custom_model_params[start:end]

    for local_index, config in enumerate(modelos_testar):
        model_index = start + local_index  # 🔹 Índice global, evita sobrescrita

        model_type = config['model']
        global POOLING
        POOLING = config['pooling']

        print(f"\n🔍 Testando modelo extrator: {model_type}...")
        print(f"[INFO] Configuração: {config}")

        start_model_type = time.time()

        # Extração de features e labels
        features, labels = feature_model_extract(df, model_type)

        skf = StratifiedKFold(n_splits=splits, shuffle=True, random_state=SEED)
        splits_list = list(skf.split(features, labels))

        # 🔄 Lista local para resultados deste modelo
        model_results = []

        for fold, (train_idx, test_idx) in enumerate(splits_list, start=1):
            print("*"*30)
            print(f"🔍 Testando modelo extrator: {model_type}...")
            print(f"[INFO] Configuração: {config}")
            print(f"[INFO] Model Index: {model_index}")
            print(f"📂 Fold: {fold}")

            X_train, X_test = features[train_idx], features[test_idx]
            y_train, y_test = labels[train_idx], labels[test_idx]

            # Parâmetros do modelo
            model_params = {
                k: v for k, v in config.items()
                if k in ['activation', 'dropout_rate', 'learning_rate', 'n_layers', 'n_neurons',
                         'optimizer', 'f1_loss_used', 'f1_alpha', 'f1_beta', 'peso_penalty_fn']
            }

            epochs = config['epochs']
            batch_size = config['batch_size']
            early_pat = config['earlystop_patience']
            reduce_factor = config['reduceLR_factor']
            reduce_pat = config['reduceLR_patience']

            try:
                model = get_classifier_model(input_shape=X_train.shape[1], **model_params)

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

                print(f"[✔] Modelo {model_index} | Fold {fold} | Tempo: {elapsed / 60:.2f} min")

                model_results.append({
                    "Fold": fold,
                    "ExtractModel": model_type,
                    "Pooling": POOLING,
                    "Model_Parameters": model_params,
                    "Epochs": epochs,
                    "batch_size": batch_size,
                    "earlystop_patience": early_pat,
                    "reduceLR_factor": reduce_factor,
                    "reduceLR_patience": reduce_pat,
                    "loss_function": 'F1_Loss' if config['f1_loss_used'] else 'BinaryCrossentropy',
                    'f1_alpha': config['f1_alpha'],
                    'f1_beta': config['f1_beta'],
                    'peso_penalty_fn': config['peso_penalty_fn'],
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred),
                    "Recall": recall_score(y_test, y_pred),
                    "F1-Score": f1_score(y_test, y_pred),
                    "ROC-AUC": roc_auc_score(y_test, y_pred)
                })

            except Exception as e:
                print(f"❌ Erro: {e}")
                with open("error_log.txt", "a") as f:
                    f.write(f"Erro no modelo {model_index}, fold {fold}: {e}\n")

            # Liberação de memória
            try:
                del model, y_pred, callbacks, X_train, X_test, y_train, y_test
            except:
                pass
            K.clear_session()
            gc.collect()

        # ✅ Agora fora do loop de folds
        df_model = pd.DataFrame(model_results)
        partial_name = f"temp_results_model{model_index}.csv"
        df_model.to_csv(partial_name, index=False)
        print(f"💾 Resultados parciais salvos para modelo {model_index}: {partial_name}")

        # Acumula resultados
        results.extend(model_results)


def merge_partial_results(path="./", output_name="CrossVal_CombinedResults.csv"):
    import glob
    partial_files = sorted(glob.glob(os.path.join(path, "temp_results_model*.csv")))

    if not partial_files:
        print("⚠️ Nenhum arquivo de resultado temporário encontrado.")
        return

    dfs = [pd.read_csv(f) for f in partial_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(output_name, index=False)
    print(f"✅ Arquivo final salvo como: {output_name}")

    # 🔥 Remover arquivos parciais após merge
    for file in partial_files:
        try:
            os.remove(file)
            print(f"🗑️ Arquivo removido: {file}")
        except Exception as e:
            print(f"⚠️ Erro ao remover {file}: {e}")

def run_chunk(custom_model_params_completo, start, end):
    configurar_gpu_para_processos(paralelos=4, reserva_mb=2048)

    # Parâmetros gerais
    features_test_size = 0.2
    features_validation_size = 0.2
    splits = 10

    msg_inicio = f"🚀 Iniciando chunk de {start} até {end}"
    log_msg(msg_inicio)

    run_CrossValidation(
        custom_model_params=custom_model_params_completo,
        splits=splits,
        features_validation_size=features_validation_size,
        start=start,
        end=end
    )

    msg_fim = f"[INFO] Chunk de {start} a {end} finalizado com sucesso ✅"
    log_msg(msg_fim)

def log_msg(msg, log_file="log_execucao_chunks.txt"):
    print(msg)  # mostra na tela
    with open(log_file, "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {msg}\n")

if __name__ == "__main__":

    start_time_total = time.time()

    #controle de compatibilidade windows/linux - multiprocessamento
    multiprocessing.set_start_method('spawn')

    #modelos selecionados no Test0004
    #
    custom_model_params = [
    #01    {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.1, 'learning_rate': 0.0005, 'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam', 'epochs': 100, 'batch_size': 64, 'earlystop_patience': 200, 'reduceLR_factor': 0.5, 'reduceLR_patience': 10},
    #02    {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.1, 'learning_rate': 0.0005, 'n_layers': 2, 'n_neurons': 128, 'optimizer': 'adam', 'epochs': 100, 'batch_size': 32, 'earlystop_patience': 100, 'reduceLR_factor': 0.3, 'reduceLR_patience': 10},
    #03     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.1, 'learning_rate': 0.0005, 'n_layers': 2, 'n_neurons': 256, 'optimizer': 'adam', 'epochs': 100, 'batch_size': 64, 'earlystop_patience': 150, 'reduceLR_factor': 0.5, 'reduceLR_patience': 15},
    #04     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.1, 'learning_rate': 0.001, 'n_layers': 1, 'n_neurons': 256, 'optimizer': 'adam', 'epochs': 1500, 'batch_size': 16, 'earlystop_patience': 150, 'reduceLR_factor': 0.5, 'reduceLR_patience': 10},
    #05     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 128, 'optimizer': 'adam', 'epochs': 1000, 'batch_size': 16, 'earlystop_patience': 100, 'reduceLR_factor': 0.3, 'reduceLR_patience': 5},
    #06     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 128, 'optimizer': 'adam', 'epochs': 100, 'batch_size': 32, 'earlystop_patience': 50, 'reduceLR_factor': 0.5, 'reduceLR_patience': 15},
    #07     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam', 'epochs': 100, 'batch_size': 16, 'earlystop_patience': 50, 'reduceLR_factor': 0.3, 'reduceLR_patience': 5},
    #08     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam', 'epochs': 250, 'batch_size': 64, 'earlystop_patience': 200, 'reduceLR_factor': 0.3, 'reduceLR_patience': 15},
    #09    {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam', 'epochs': 500, 'batch_size': 64, 'earlystop_patience': 150, 'reduceLR_factor': 0.3, 'reduceLR_patience': 10},
    #10     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam', 'epochs': 500, 'batch_size': 64, 'earlystop_patience': 150, 'reduceLR_factor': 0.3, 'reduceLR_patience': 15},
    #11     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.3, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 256, 'optimizer': 'adam', 'epochs': 500, 'batch_size': 16, 'earlystop_patience': 50, 'reduceLR_factor': 0.3, 'reduceLR_patience': 15},
    #12     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.3, 'learning_rate': 0.0005, 'n_layers': 1, 'n_neurons': 128, 'optimizer': 'adam', 'epochs': 500, 'batch_size': 64, 'earlystop_patience': 200, 'reduceLR_factor': 0.5, 'reduceLR_patience': 10},
    #13     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.4, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam', 'epochs': 100, 'batch_size': 32, 'earlystop_patience': 200, 'reduceLR_factor': 0.5, 'reduceLR_patience': 5},
    #14     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.4, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam', 'epochs': 250, 'batch_size': 32, 'earlystop_patience': 200, 'reduceLR_factor': 0.3, 'reduceLR_patience': 5},
    #15    {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.4, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam', 'epochs': 500, 'batch_size': 64, 'earlystop_patience': 150, 'reduceLR_factor': 0.3, 'reduceLR_patience': 10},
     #16   {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.4, 'learning_rate': 0.001, 'n_layers': 1, 'n_neurons': 256, 'optimizer': 'adam', 'epochs': 250, 'batch_size': 32, 'earlystop_patience': 50, 'reduceLR_factor': 0.5, 'reduceLR_patience': 10}
     ]

    custom_model_params = [
    #01    {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.1, 'learning_rate': 0.0005, 'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam', 'epochs': 100, 'batch_size': 64, 'earlystop_patience': 200, 'reduceLR_factor': 0.5, 'reduceLR_patience': 10},
    #02    {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.1, 'learning_rate': 0.0005, 'n_layers': 2, 'n_neurons': 128, 'optimizer': 'adam', 'epochs': 100, 'batch_size': 32, 'earlystop_patience': 100, 'reduceLR_factor': 0.3, 'reduceLR_patience': 10},
    #03     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.1, 'learning_rate': 0.0005, 'n_layers': 2, 'n_neurons': 256, 'optimizer': 'adam', 'epochs': 100, 'batch_size': 64, 'earlystop_patience': 150, 'reduceLR_factor': 0.5, 'reduceLR_patience': 15},
    #04     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.1, 'learning_rate': 0.001, 'n_layers': 1, 'n_neurons': 256, 'optimizer': 'adam', 'epochs': 1500, 'batch_size': 16, 'earlystop_patience': 150, 'reduceLR_factor': 0.5, 'reduceLR_patience': 10},
    #05     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 128, 'optimizer': 'adam', 'epochs': 1000, 'batch_size': 16, 'earlystop_patience': 100, 'reduceLR_factor': 0.3, 'reduceLR_patience': 5},
    #06     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 128, 'optimizer': 'adam', 'epochs': 100, 'batch_size': 32, 'earlystop_patience': 50, 'reduceLR_factor': 0.5, 'reduceLR_patience': 15},
    #07     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam', 'epochs': 100, 'batch_size': 16, 'earlystop_patience': 50, 'reduceLR_factor': 0.3, 'reduceLR_patience': 5},
    #08     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam', 'epochs': 250, 'batch_size': 64, 'earlystop_patience': 200, 'reduceLR_factor': 0.3, 'reduceLR_patience': 15},
    #09    {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam', 'epochs': 500, 'batch_size': 64, 'earlystop_patience': 150, 'reduceLR_factor': 0.3, 'reduceLR_patience': 10},
    #10     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.2, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam', 'epochs': 500, 'batch_size': 64, 'earlystop_patience': 150, 'reduceLR_factor': 0.3, 'reduceLR_patience': 15},
    #11     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.3, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 256, 'optimizer': 'adam', 'epochs': 500, 'batch_size': 16, 'earlystop_patience': 50, 'reduceLR_factor': 0.3, 'reduceLR_patience': 15},
    #12     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.3, 'learning_rate': 0.0005, 'n_layers': 1, 'n_neurons': 128, 'optimizer': 'adam', 'epochs': 500, 'batch_size': 64, 'earlystop_patience': 200, 'reduceLR_factor': 0.5, 'reduceLR_patience': 10},
    #13     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.4, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam', 'epochs': 100, 'batch_size': 32, 'earlystop_patience': 200, 'reduceLR_factor': 0.5, 'reduceLR_patience': 5},
    #14     {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.4, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam', 'epochs': 250, 'batch_size': 32, 'earlystop_patience': 200, 'reduceLR_factor': 0.3, 'reduceLR_patience': 5},
    #15    {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.4, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam', 'epochs': 500, 'batch_size': 64, 'earlystop_patience': 150, 'reduceLR_factor': 0.3, 'reduceLR_patience': 10},
     #16   {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.4, 'learning_rate': 0.001, 'n_layers': 1, 'n_neurons': 256, 'optimizer': 'adam', 'epochs': 250, 'batch_size': 32, 'earlystop_patience': 50, 'reduceLR_factor': 0.5, 'reduceLR_patience': 10}
     ]


    # Beta
    # 0.3 --> Precisão extrema - Evita falsos positivos
    # 0.5 --> Alta precisão - Problemas com alto custo para FP
    # 1.0 --> Equilíbrio - Tradicional F1 - Score
    # 1.5 --> Foco leve em recall - Compromisso razoável
    # 2.0 --> Recall dominante - Problemascom alto custo para FN
    # 3.0 --> Recall extremo - Quando não detectar e critico

    optimizion_params = [
        {'f1_loss_used': True, 'f1_alpha': 1.0, 'f1_beta': 3.0, 'peso_penalty_fn': 2},
        {'f1_loss_used': True, 'f1_alpha': 1.0, 'f1_beta': 2.5, 'peso_penalty_fn': 2},
        {'f1_loss_used': True, 'f1_alpha': 1.0, 'f1_beta': 2.0, 'peso_penalty_fn': 2},
        {'f1_loss_used': True, 'f1_alpha': 0.8, 'f1_beta': 3.0, 'peso_penalty_fn': 2},
        {'f1_loss_used': True, 'f1_alpha': 1.0, 'f1_beta': 1.5, 'peso_penalty_fn': 2},
        {'f1_loss_used': True, 'f1_alpha': 0.7, 'f1_beta': 1.5, 'peso_penalty_fn': 2},
        {'f1_loss_used': True, 'f1_alpha': 0.5, 'f1_beta': 2.0, 'peso_penalty_fn': 2},
        {'f1_loss_used': True, 'f1_alpha': 0.8, 'f1_beta': 1.8, 'peso_penalty_fn': 2},
        {'f1_loss_used': True, 'f1_alpha': 0.8, 'f1_beta': 1.0, 'peso_penalty_fn': 2},
        {'f1_loss_used': True, 'f1_alpha': 0.6, 'f1_beta': 2.0, 'peso_penalty_fn': 2},
        {'f1_loss_used': True, 'f1_alpha': 0.8, 'f1_beta': 2.0, 'peso_penalty_fn': 2},
        {'f1_loss_used': True, 'f1_alpha': 1.0, 'f1_beta': 2.0, 'peso_penalty_fn': 2},
        {'f1_loss_used': True, 'f1_alpha': 0.7, 'f1_beta': 2.0, 'peso_penalty_fn': 2},
        {'f1_loss_used': True, 'f1_alpha': 1.0, 'f1_beta': 3.0, 'peso_penalty_fn': 1.5},
        {'f1_loss_used': True, 'f1_alpha': 1.0, 'f1_beta': 2.5, 'peso_penalty_fn': 1.5},
        {'f1_loss_used': True, 'f1_alpha': 1.0, 'f1_beta': 2.0, 'peso_penalty_fn': 1.5},
        {'f1_loss_used': True, 'f1_alpha': 0.8, 'f1_beta': 3.0, 'peso_penalty_fn': 1.5},
        {'f1_loss_used': True, 'f1_alpha': 1.0, 'f1_beta': 1.5, 'peso_penalty_fn': 1.5},
        {'f1_loss_used': True, 'f1_alpha': 0.7, 'f1_beta': 1.5, 'peso_penalty_fn': 1.5},
        {'f1_loss_used': True, 'f1_alpha': 0.5, 'f1_beta': 2.0, 'peso_penalty_fn': 1.5},
        {'f1_loss_used': True, 'f1_alpha': 0.8, 'f1_beta': 1.8, 'peso_penalty_fn': 1.5},
        {'f1_loss_used': True, 'f1_alpha': 0.8, 'f1_beta': 1.0, 'peso_penalty_fn': 1.5},
        {'f1_loss_used': True, 'f1_alpha': 0.6, 'f1_beta': 2.0, 'peso_penalty_fn': 1.5},
        {'f1_loss_used': True, 'f1_alpha': 0.8, 'f1_beta': 2.0, 'peso_penalty_fn': 1.5},
        {'f1_loss_used': True, 'f1_alpha': 1.0, 'f1_beta': 2.0, 'peso_penalty_fn': 1.5},
        {'f1_loss_used': True, 'f1_alpha': 0.7, 'f1_beta': 2.0, 'peso_penalty_fn': 1.5},
    ]


    # Combinação completa
    custom_model_params_completo = [
        {**base, **opt}
        for base in custom_model_params
        for opt in optimizion_params
    ]

    total = len(custom_model_params_completo)
    n_threads = 2
    chunk_size = math.ceil(total / n_threads)  # 🔹 cálculo dinâmico

    chunks = [(i, min(i + chunk_size, total)) for i in range(0, total, chunk_size)]

    for i in range(0, len(chunks), n_threads):  # Executa 4 chunks por vez
        chunk_pair = chunks[i:i + n_threads]
        processes = []
        for start, end in chunk_pair:
            p = multiprocessing.Process(target=run_chunk, args=(custom_model_params_completo, start, end))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

    print("✅ Todos os chunks foram processados.")

    #partial files - Combined Results
    merge_partial_results()

    elapsed_total = time.time() - start_time_total

    print(f"[INFO] Tempo Total de processamento: {elapsed_total / 60:.2f} minutos")




