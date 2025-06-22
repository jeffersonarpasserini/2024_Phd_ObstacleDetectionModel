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
import matplotlib.pyplot as plt
import seaborn as sns

from itertools import product

from scipy.optimize import minimize_scalar
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import f1_score, fbeta_score, recall_score, precision_score, matthews_corrcoef, accuracy_score
from sklearn.metrics import roc_auc_score, confusion_matrix
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
os.environ['TF_DETERMINISTIC_OPS'] = '1'
NUM_PROCESSES = 4  # Limitar explicitamente o número de folds simultâneos por GPU

# Garantir reprodutibilidade
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

#sem uso
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
    # 🔄 Corrigido: arredonda e limita de 0 a 1
    # k.round() assume entrada entre 0 e 1 se o modelo gerar algo fora da faixa pode ocasionar comportamente inesperado
    # K.clip() garante 0 ou  1 com entrada

    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

    return f1

#função de perda personalizada
def get_f1_loss(alpha=0.6, beta=1.0):
    def f1_loss(y_true, y_pred):

        # alpha
        # Usa o parâmetro alpha para controlar dinamicamente o peso entre F1 e precisão.
        # alpha = 0.6 --> resulta em 1 - (f1 * 0.6 + precision * 0.4)
        # 60% do peso f1 e 40% precisao

        # beta
        # Beta > 1 - favorece o recall (util se falsos negativos sao piores para o problema)
        # Beta < 1 - favorece a precisao (util se falsos positivos sao piores)
        # Beta = 1 - formula tradicional do F1-Score

        y_true = K.cast(y_true, 'float32')
        tp = K.sum(y_true * y_pred)
        fp = K.sum((1 - y_true) * y_pred)
        fn = K.sum(y_true * (1 - y_pred))

        precision = tp / (tp + fp + K.epsilon())
        recall = tp / (tp + fn + K.epsilon())

        f1 = (1 + beta ** 2) * (precision * recall) / (beta ** 2 * precision + recall + K.epsilon())

        return 1 - (alpha * f1 + (1 - alpha) * precision)

    return f1_loss

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
                         n_layers=1, n_neurons=128, optimizer='adam', f1_loss_used=False, f1_alpha=0.6, f1_beta=1.0):

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
        loss_fn = get_f1_loss(alpha=f1_alpha, beta=f1_beta)
        model.compile(optimizer=optimizer_instance, loss=loss_fn, metrics=['accuracy', f1_metric])
    else:
        model.compile(optimizer=optimizer_instance, loss=BinaryCrossentropy(), metrics=['accuracy'])

    return model

def executar_fold_em_subprocesso(fold_args):
    (fold, train_idx, test_idx, features, labels, config, model_index, model_type, POOLING, features_validation_size, image_name) = fold_args

    import os
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Evita alocação total
    configurar_gpu_para_processos(paralelos=NUM_PROCESSES, reserva_mb=2048)

    import tensorflow as tf
    from tensorflow.keras import backend as K

    try:
        start_img_time = time.time()

        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        model_params = {
            k: v for k, v in config.items()
            if k in ['activation', 'dropout_rate', 'learning_rate', 'n_layers', 'n_neurons',
                     'optimizer', 'f1_loss_used', 'f1_alpha', 'f1_beta']
        }

        epochs = config['epochs']
        batch_size = config['batch_size']
        early_pat = config['earlystop_patience']
        reduce_factor = config['reduceLR_factor']
        reduce_pat = config['reduceLR_patience']

        # limpa a memoria GPU para o tensorflow
        gc.collect()
        K.clear_session()
        tf.compat.v1.reset_default_graph()

        model = get_classifier_model(input_shape=X_train.shape[1], **model_params)

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=early_pat, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=reduce_factor, patience=reduce_pat, min_lr=1e-5)
        ]

        model.fit(X_train, y_train,
                  validation_split=features_validation_size,
                  epochs=epochs,
                  batch_size=batch_size,
                  callbacks=callbacks,
                  verbose=0)

        y_pred_prob = float(model.predict(X_test, batch_size=1, verbose=0).squeeze())
        y_pred = int(y_pred_prob > 0.5) #0.5 threshold default

        tn, fp, fn, tp = confusion_matrix([y_test], [y_pred], labels=[0, 1]).ravel()

        resultado = {
            "Fold": fold,
            "Image_Name": image_name,
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
            "TP": int(tp),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "y_pred_prob": float(y_pred_prob),
            "y_test": int(y_test)
        }

        print(f"Fold:{fold} - Pred:{y_pred_prob}-{y_pred}")
        end_img_time = time.time()
        img_elapsed_time = end_img_time - start_img_time
        print(f"[INFO] Tempo Total de processamento: {img_elapsed_time / 60:.2f} minutos")

        # limpeza segura apenas de variáveis existentes
        for var in ['model', 'X_train', 'X_test', 'y_train', 'y_test',
                    'y_pred_prob', 'y_pred', 'callbacks', 'model_params']:
            if var in locals():
                del locals()[var]

        gc.collect()
        K.clear_session()
        return resultado

    except Exception as e:
        print(f"❌ Erro no fold {fold}: {e}")
        # limpeza segura apenas de variáveis existentes
        for var in ['model', 'X_train', 'X_test', 'y_train', 'y_test',
                    'y_pred_prob', 'y_pred', 'callbacks', 'model_params']:
            if var in locals():
                del locals()[var]

        gc.collect()
        K.clear_session()
        return None

def run_LOOCV(custom_model_params, model_start=0, model_end=None, features_validation_size=0.2, fold_start=0, fold_end=None):
    print("\n=========================== LOOCV (Leave-One-Out) ===========================")
    df = load_data()
    results = []

    modelos_testar = custom_model_params[model_start:model_end]

    for local_index, config in enumerate(modelos_testar):
        model_index = model_start + local_index
        model_type = config['model']
        global POOLING
        POOLING = config['pooling']

        print(f"\n🔍 Testando modelo extrator (LOOCV): {model_type}...")
        print(f"[INFO] Configuração: {config}")

        features, labels = feature_model_extract(df, model_type)
        loo = LeaveOneOut()
        splits_list = list(loo.split(features, labels))
        splits_list = splits_list[fold_start:fold_end]
        total_folds = len(splits_list)

        # Lista de nomes dos arquivos de imagem
        image_names = df["filename"].tolist()

        args_para_folds = [
            (fold, train_idx, test_idx, features, labels, config, model_index, model_type, POOLING, features_validation_size,
              os.path.basename(image_names[test_idx[0]]))
            for fold, (train_idx, test_idx) in enumerate(splits_list, start=1)
        ]

        print(f"📦 Executando {total_folds} folds em subprocessos com limite de {NUM_PROCESSES} simultâneos...")
        with multiprocessing.Pool(processes=NUM_PROCESSES) as pool:
            model_results = list(tqdm(
                                    pool.imap(executar_fold_em_subprocesso, args_para_folds),
                                    total=len(args_para_folds),desc=f"Modelo {model_index} - LOOCV"))

        model_results = [r for r in model_results if r is not None]

        # Salvando resultados do chunk
        chunk_filename = f"./temp_results_loocv_model{model_index}_folds_{fold_start}_{fold_end}.csv"
        df_chunk = pd.DataFrame(model_results)
        df_chunk.to_csv(chunk_filename, index=False)
        print(f"[INFO] Chunk de model {model_start} a {model_end} e fold: {fold_start} a {fold_end} finalizado e salvo em: {chunk_filename}")

    return results

def analisar_resultados_completos(model_index):

    summary_results = []

    temp_files = [f for f in os.listdir('.') if f.startswith(f'temp_results_loocv_model{model_index}_folds') and f.endswith('.csv')]
    if not temp_files:
        print("❌ Nenhum arquivo temporário encontrado.")
        return

    print(f"📁 Encontrados {len(temp_files)} arquivos temporários.")
    df_all = pd.concat([pd.read_csv(f) for f in sorted(temp_files)])

    df_all.to_csv(f"LOOCV_CombinedResults_model{model_index}.csv", index=False)
    print(f"✅ Arquivo final LOOCV salvo como: LOOCV_CombinedResults_model{model_index}.csv")

    print("📈 Cálculo do melhor threshold global com base em todos os y_pred_prob e y_test")
    all_probs = df_all["y_pred_prob"].to_numpy()
    all_labels = df_all["y_test"].to_numpy()

    best_thresh, all_metrics = global_best_threshold_combined(all_labels, all_probs)

    df_all['Index'] = df_all.index
    df_all['Best_Threshold_Global'] = best_thresh

    # para visualização:
    thresholds = np.array(all_metrics["thresholds"])
    f1_scores = np.array(all_metrics["f1"])
    print(
        f"✅ Threshold global ótimo: {best_thresh:.4f} com score combinado máximo: {np.max(all_metrics['combined']):.4f}")

    # Visualização e salvamento da distribuição dos thresholds
    os.makedirs("diagnostics", exist_ok=True)
    hist_path = f"diagnostics/threshold_distribution_model{model_index}.png"
    csv_path = f"diagnostics/best_thresholds_model{model_index}.csv"

    plt.figure()
    plt.hist(df_all["y_pred_prob"], bins=20, color="skyblue")
    plt.title("Distribuição das probabilidades preditas (y_pred_prob)")
    plt.xlabel("Probabilidade prevista")
    plt.ylabel("Frequência")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(hist_path)
    print(f"📊 Histograma salvo em: {hist_path}")

    pd.DataFrame({"Best_Threshold_Global": [best_thresh]}).to_csv(csv_path, index=False)
    pd.DataFrame(all_metrics).to_csv(f"diagnostics/threshold_metrics_model{model_index}.csv", index=False)
    print(f"📁 CSV de thresholds salvos em: {csv_path}")

    # 🎯 Recalcular métricas com o threshold global
    y_pred_global = (all_probs > best_thresh).astype(int)
    tn, fp, fn, tp = confusion_matrix(all_labels, y_pred_global, labels=[0, 1]).ravel()

    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    npv = tn / (tn + fn + 1e-8)
    mcc_numerator = (tp * tn) - (fp * fn)
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-8
    mcc = mcc_numerator / mcc_denominator

    # 🧾 Salvar curva F1-score vs Threshold
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, all_metrics["f2"], label="F2-score")
    plt.plot(thresholds, all_metrics["recall"], label="Recall")
    plt.plot(thresholds, all_metrics["mcc"], label="MCC")
    plt.plot(thresholds, all_metrics["precision"], label="Precision")
    plt.axvline(x=best_thresh, color='red', linestyle='--', label=f'Threshold ótimo: {best_thresh:.2f}')
    plt.title("Curvas de Métricas por Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"diagnostics/combined_metrics_model{model_index}.png")

    summary_results.append({
        "Model_Index": model_index,
        "ExtractModel": df_all["ExtractModel"].iloc[0],
        "Pooling": df_all["Pooling"].iloc[0],
        "Model_Parameters": df_all["Model_Parameters"].iloc[0],
        "Epochs": df_all["Epochs"].iloc[0],
        "batch_size": df_all["batch_size"].iloc[0],
        "earlystop_patience": df_all["earlystop_patience"].iloc[0],
        "reduceLR_factor": df_all["reduceLR_factor"].iloc[0],
        "reduceLR_patience": df_all["reduceLR_patience"].iloc[0],
        "loss_function": df_all["loss_function"].iloc[0] if 'loss_function' in df_all.columns else 'Unknown',
        "f1_alpha": df_all["f1_alpha"].iloc[0],
        "f1_beta": df_all["f1_beta"].iloc[0],
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "Specificity": specificity,
        "NPV": npv,
        "MCC": mcc,
        "Median_Threshold_wo_Outliers": best_thresh
    })

    pd.DataFrame(summary_results).to_csv(f"summary_results_loocv_model{model_index}.csv", index=False)
    print(f"✅ Resultados agregados salvos em summary_results_loocv_model{model_index}.csv")

    # Distribuição dos thresholds
    plt.figure(figsize=(10, 6))
    sns.histplot(df_all['y_pred_prob'], bins=30, kde=True)
    plt.title('Distribuição dos Thresholds - Modelo {}'.format(model_index))
    plt.xlabel('Threshold')
    plt.ylabel('Frequência')
    os.makedirs("diagnostics", exist_ok=True)
    plt.savefig(f"diagnostics/threshold_distribution_model{model_index}.png")
    print(f"📊 Histograma salvo em: diagnostics/threshold_distribution_model{model_index}_novo.png")

    # Salvar CSV com thresholds
    df_all[['Index', 'Best_Threshold_Global']].to_csv(f"diagnostics/best_thresholds_model{model_index}_novo.csv", index=False)
    print(f"📁 CSV de thresholds salvos em: diagnostics/best_thresholds_model{model_index}_novo.csv")

    # Calcular média do melhor threshold
    best_thresh = df_all['Best_Threshold_Global'].median()

    # F1 x Threshold plot
    thresholds = sorted(df_all['Best_Threshold_Global'].unique())
    f1_scores = []
    for t in thresholds:
        preds = (df_all['y_pred_prob'] >= t).astype(int)
        f1 = f1_score(df_all['y_test'], preds)
        f1_scores.append(f1)

    df_f1 = pd.DataFrame({"Threshold": thresholds, "F1_Score": f1_scores})
    df_f1.to_csv(f"diagnostics/f1_scores_by_threshold_model{model_index}_novo.csv", index=False)
    print(f"📁 CSV de F1-scores por threshold salvo em: diagnostics/f1_scores_by_threshold_model{model_index}_novo.csv")

    plt.figure(figsize=(10, 6))
    plt.plot(df_f1['Threshold'], df_f1['F1_Score'], marker='o')
    plt.axvline(x=best_thresh, color='red', linestyle='--', label=f'Threshold Médio: {best_thresh:.2f}')
    plt.title(f'F1-score vs Threshold - Modelo {model_index}')
    plt.xlabel('Threshold')
    plt.ylabel('F1-score')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"diagnostics/f1_curve_model{model_index}_novo.png")
    print(f"📈 Curva F1-score salva em: diagnostics/f1_curve_model{model_index}_novo.png")

    print("✅ Análise final concluída com sucesso!")

def global_best_threshold_combined(y_true, y_probs, beta=2.0):
    thresholds = np.linspace(0.1, 0.9, 200)

    all_metrics = {
        "thresholds": [],
        "f1": [],
        "f2": [],
        "recall": [],
        "recall_class0": [],
        "precision": [],
        "mcc": [],
        "accuracy": [],
        "combined": []
    }

    for t in thresholds:
        y_pred = (y_probs > t).astype(int)

        f1 = f1_score(y_true, y_pred, zero_division=0)
        f2 = fbeta_score(y_true, y_pred, beta=beta, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)  # recall da classe 1
        precision = precision_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)

        # 📌 Recall da classe 0 (obstacle)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
        recall_class0 = tn / (tn + fp + 1e-8)  # Specificity para classe 1 == recall da classe 0

        # 🎯 Score combinado com mais peso para recall da classe 1
        combined = 0.5 * recall + 0.3 * f2 + 0.1 * mcc + 0.1 * precision

        all_metrics["thresholds"].append(t)
        all_metrics["f1"].append(f1)
        all_metrics["f2"].append(f2)
        all_metrics["recall"].append(recall)
        all_metrics["recall_class0"].append(recall_class0)
        all_metrics["precision"].append(precision)
        all_metrics["mcc"].append(mcc)
        all_metrics["accuracy"].append(acc)
        all_metrics["combined"].append(combined)

    best_score = max(all_metrics["combined"])
    best_threshold_candidates = [t for t, score in zip(thresholds, all_metrics["combined"]) if score == best_score]
    best_threshold = max(best_threshold_candidates)

    print(f"🎯 Melhor threshold: {best_threshold:.4f} com score combinado: {best_score:.4f}")
    return best_threshold, all_metrics


def run_chunk(custom_model_params_completo, model_start, model_end, log_file_name, fold_start, fold_end):

    features_validation_size = 0.2  # usado apenas na validação LOOCV

    msg_inicio = f"🚀 Iniciando chunk de {model_start} até {model_end} e dataset de {fold_start} até {fold_end}"
    log_msg(msg_inicio, log_file_name)

    run_LOOCV(
        custom_model_params=custom_model_params_completo,
        model_start=model_start,
        model_end=model_end,
        features_validation_size=features_validation_size,
        fold_start=fold_start,
        fold_end=fold_end
    )

    msg_fim = f"[INFO] Chunk de {model_start} a {model_end} e dataset de {fold_start} até {fold_end} finalizado com sucesso ✅"
    log_msg(msg_fim, log_file_name)

def log_msg(msg, log_file):
    print(msg)  # mostra na tela
    with open(log_file, "a", encoding="utf-8") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {msg}\n")

if __name__ == "__main__":

    start_time_total = time.time()

    executa = True

    if (executa):
        #controle de compatibilidade windows/linux - multiprocessamento
        multiprocessing.set_start_method('spawn')

        log_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file_name = f"log_execucao_chunks_{log_timestamp}.txt"

        # Modelos selecionados test0005
        custom_model_params_completo = [

            # modelo 01
                #("Test0004 | MobileNetV1_avg_{'activation': 'relu', 'dropout_rate': 0.1, 'learning_rate': 0.0005, 'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam'}"
                #"_100_64_200_0.5_10"), 2544.1, Test0004, 1566.05, 1750.5, 1953.55
               {'model': 'MobileNetV1', 'pooling': 'avg', 'activation': 'relu', 'dropout_rate': 0.1,
                'learning_rate': 0.0005,
                'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam', 'epochs': 100, 'batch_size': 64,
                'earlystop_patience': 200, 'reduceLR_factor': 0.5, 'reduceLR_patience': 10, 'f1_loss_used': False,
                'f1_alpha': 0, 'f1_beta': 0},

            #model 02
            # "Test0004 | MobileNetV1_avg_{'activation': 'relu', 'dropout_rate': 0.1, 'learning_rate': 0.0005, 'n_layers': 2, 'n_neurons': 128, 'optimizer': 'adam'}_100_32_100_0.3_10",
            # 2493.05, Test0004, 1532.7, 1875.7, 1967.1499999999999

            #model 03
            # "Test0004 | MobileNetV1_avg_{'activation': 'relu', 'dropout_rate': 0.4, 'learning_rate': 0.001, 'n_layers': 1, 'n_neurons': 256, 'optimizer': 'adam'}_250_32_50_0.5_10",
            # 2574.35, Test0004, 1618.45, 1836.4, 2009.7333333333336

            #model 04
            # "Test0004 | MobileNetV1_avg_{'activation': 'relu', 'dropout_rate': 0.4, 'learning_rate': 0.0001, 'n_layers': 1, 'n_neurons': 512, 'optimizer': 'adam'}_500_64_150_0.3_10",
            # 1558.7, Test0004, 2923.1, 1592.2, 2024.6666666666667

            #model 05
            # "Test0004 | MobileNetV1_avg_{'activation': 'relu', 'dropout_rate': 0.1, 'learning_rate': 0.001, 'n_layers': 1, 'n_neurons': 256, 'optimizer': 'adam'}_1500_16_150_0.5_10",
            # 2033.65, Test0004, 2489.25, 1573.4, 2032.0999999999997

        ]

        total = len(custom_model_params_completo)
        n_threads = 1
        chunk_size = math.ceil(total / n_threads)  # 🔹 cálculo dinâmico

        chunks = [(i, min(i + chunk_size, total)) for i in range(0, total, chunk_size)]

        fold_splits = [(0, 500), (500, 1000), (1000, 1500), (1500, 1929)]

        for fold_start, fold_end in fold_splits:
            for i in range(0, len(chunks), n_threads):  # Executa n chunks por vez
                chunk_pair = chunks[i:i + n_threads]
                processes = []
                for model_start, model_end in chunk_pair:
                    p = multiprocessing.Process(target=run_chunk, args=(custom_model_params_completo, model_start, model_end, log_file_name, fold_start, fold_end))
                    p.start()
                    processes.append(p)

            for p in processes:
                p.join()

        print("✅ Todos os chunks foram processados.")

    analisar_resultados_completos(model_index=0)

    elapsed_total = time.time() - start_time_total

    print(f"[INFO] Tempo Total de processamento: {elapsed_total / 60:.2f} minutos")




