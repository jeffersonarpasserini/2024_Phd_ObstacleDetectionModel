import datetime
import os
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
import random
import shap

from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from keras_preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
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

# Dataset's
DATASET_VIA_DATASET = os.path.join(BASE_PATH, '..', 'via-dataset')
DATASET_VIA_DATASET_EXTENDED = os.path.join(BASE_PATH, '..', 'via-dataset-extended')

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
        #class_mode='categorical',
        class_mode='binary' if len(df['category'].unique()) == 2 else 'categorical',
        target_size=IMAGE_SIZE,
        batch_size=batch_size,
        shuffle=False  # originalmente estava False --> agora com True aplica a aleatorização das imagens
    )

    features = model.predict(generator, steps=steps)
    return features

def feature_model_extract(df, model_type, use_augmentation=False, use_shap=False, sample_size=None):
    """Executa o processo de extração de características, podendo incluir Data Augmentation e SHAP."""
    model, preprocessing_function = get_extract_model(model_type)

    features = extract_features(df, model, preprocessing_function, use_augmentation)  # ✅ Passar o parâmetro

    labels = df["category"].replace({'clear': 1, 'non-clear': 0}).to_numpy().astype(int)  # 🔹 Adicionar labels corretamente

    if use_shap:
        analyze_shap(model, df, preprocessing_function, sample_size)

    return features


def analyze_shap(model, df, preprocessing_function, sample_size=None):
    """Executa análise SHAP para visualizar importância das características nas predições do classificador."""
    if sample_size is None:
        sample_size = int(0.05 * len(df))  # 5% do total
    sample_size = max(50, min(sample_size, 500))  # Garante que está dentro dos limites

    print(f"🔍 Executando SHAP com {sample_size} imagens.")

    # Amostragem de imagens
    sample_df = df.sample(n=sample_size, random_state=SEED)

    # Carregar imagens originais para SHAP
    image_paths = [os.path.join(DATASET_PATH, fname) for fname in sample_df["filename"]]
    sample_images = np.array([preprocess_image(img_path, preprocessing_function) for img_path in image_paths])

    # Certifique-se de que sample_images tem a forma correta
    print(f"📌 Sample Images Shape: {sample_images.shape}")  # Deve ser (sample_size, 224, 224, 3)

    # Criar um conjunto de imagens de referência para o SHAP
    background = sample_images[:10]  # Usa as primeiras 10 imagens como referência

    print("Iniciando cálculo SHAP...")
    # Criar o explicador baseado no Gradiente SHAP
    explainer = shap.GradientExplainer(model, background)
    print("Explainer criado com sucesso...")

    print(f"⚙️ Calculando SHAP para {sample_images.shape[0]} imagens...")
    # Calcular valores SHAP
    shap_values = explainer.shap_values(sample_images)
    print("✅ Cálculo SHAP concluído!")

    print("📊 Gerando gráfico SHAP...")
    # Plotar gráfico SHAP
    shap.image_plot(shap_values, sample_images)
    save_path = os.path.join(BASE_PATH, "shap_summary_plot.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"📊 Gráfico SHAP salvo em: {save_path}")

    print("✅ Processo SHAP finalizado!")
    plt.close()

def preprocess_image(img_path, preprocessing_function):
    """Carrega e pré-processa uma imagem para o modelo MobileNet."""
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adiciona dimensão batch
    img_array = preprocessing_function(img_array)  # Aplica a função de pré-processamento do modelo
    return img_array[0]  # Remove a dimensão extra

# -------------------------------- CLASSIFICADOR  ---------------------------------------------------------

def get_classifier_model(input_shape, activation, dropout_rate, learning_rate, n_layers, n_neurons):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    for _ in range(n_layers):
        model.add(Dense(n_neurons, activation=activation))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    optimizer = RMSprop(learning_rate=learning_rate)
    # model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=['accuracy'])
    # Compilação do modelo com a nova métrica F1
    model.compile(optimizer=optimizer, loss=f1_loss, metrics=['accuracy', f1_metric])

    return model

# -------- Calculos para o threshold dinamico
def find_best_threshold(y_test, y_pred_prob):

    y_pred_prob = np.array(y_pred_prob)
    y_test = np.array(y_test)

    def neg_f1(thresh):
        y_pred = (y_pred_prob > thresh).astype("int32")
        return -f1_score(np.array(y_test).ravel(), np.array(y_pred).ravel(), zero_division=0)  # <- Modificação aqui

    result = minimize_scalar(neg_f1, bounds=(0.3, 0.6), method='bounded')
    return result.x if result.success else 0.6 # Threshold padrão: 0.5 --> ajustado para 0.6 testes anteriores com bons resultados

def calculate_median_threshold(df):
    thresholds = df["best_threshold"]
    Q1 = thresholds.quantile(0.25)
    Q3 = thresholds.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    filtered_thresholds = thresholds[(thresholds >= lower_bound) & (thresholds <= upper_bound)]
    median_threshold = filtered_thresholds.median()
    return median_threshold

# Define metricas para o modelo classificaro - f1_metric e f1_loss
def f1_metric(y_true, y_pred):
    """ Calcula o F1-Score como métrica """
    y_true = K.cast(y_true, 'float32')  # 🔹 Converte para float antes da multiplicação
    y_pred = K.round(y_pred)  # Converte probabilidades em classes binárias (0 ou 1)

    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))

    precision = tp / (tp + fp + K.epsilon())  # K.epsilon() evita divisão por zero
    recall = tp / (tp + fn + K.epsilon())

    return 2 * (precision * recall) / (precision + recall + K.epsilon())

def f1_loss(y_true, y_pred):
    """ Função de perda baseada no F1-Score """
    y_true = K.cast(y_true, 'float32')
    # y_pred = K.round(y_pred)  # Converte probabilidades para 0 ou 1

    tp = K.sum(y_true * y_pred)
    fp = K.sum((1 - y_true) * y_pred)
    fn = K.sum(y_true * (1 - y_pred))

    precision = tp / (tp + fp + K.epsilon())
    recall = tp / (tp + fn + K.epsilon())

    f1 = 2 * (precision * recall) / (precision + recall + K.epsilon())

    # 🔹 Ajustamos para dar mais peso à precisão e evitar que tudo seja classificado como positivo
    #return 1 - (f1 * 0.75 + precision * 0.25)
    # 📌 Ajustamos a função de perda para balancear melhor precisão e recall
    return 1 - (f1 * 0.6 + precision * 0.4)

# ------ Executa os testes ------

# Executa teste de validação cruzada
def run_kfold_cv(activation, dropout_rate, learning_rate, n_layers, n_neurons,
                 n_epochs, batch_size, early_stop_patience, lr_scheduler_patience,
                 model_type, n_splits=5, use_augmentation = False, use_shap = False, sample_size = None):

    warnings = []
    history_log = {"epoch": [], "val_loss": [], "val_f1_score": []}
    df = load_data()

    labels = df["category"].replace({'clear': 1, 'non-clear': 0}).to_numpy().astype(int)
    features = feature_model_extract(df, model_type, use_augmentation, use_shap, sample_size)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    results = []

    total_samples = len(df)
    print(f"Total de Amostras do Dataset {total_samples}...")

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels), 1):
        print(f"📌 Rodando Fold {fold}/{n_splits}...")

        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        model = get_classifier_model(X_train.shape[1], activation, dropout_rate, learning_rate, n_layers, n_neurons)

        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stop_patience, min_delta=0.001,
                                       restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=lr_scheduler_patience, min_lr=1e-5)

        # Divide os dados de treino para criar um conjunto de validação (80% treino / 20% validação)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=SEED)

        print(f"Amostras para Treinamento {len(X_train)}...")
        print(f"Amostras para Validação {len(X_val)}...")
        print(f"Amostras para Teste {len(X_test)}...")

        y_train = y_train.astype("float32")
        y_val = y_val.astype("float32")

        history = model.fit(X_train, y_train,
                            epochs=n_epochs,
                            batch_size=min(batch_size, len(X_train)),
                            validation_data=(X_val, y_val),
                            callbacks=[early_stopping, lr_scheduler],
                            verbose=0)

        # Salvando métricas por época
        for epoch, (vloss, vf1) in enumerate(zip(history.history["val_loss"], history.history["val_f1_metric"])):
            if epoch >= len(history_log["epoch"]):
                history_log["epoch"].append(epoch)
                history_log["val_loss"].append([])
                history_log["val_f1_score"].append([])

            history_log["val_loss"][epoch].append(vloss)
            history_log["val_f1_score"][epoch].append(vf1)

        y_pred_prob = model.predict(X_test).flatten()
        best_threshold = find_best_threshold(y_test, y_pred_prob)

        y_pred = (y_pred_prob > best_threshold).astype("int32")
        cm = confusion_matrix(y_test, y_pred)

        tn, fp, fn, tp = cm.flatten() if cm.shape == (2, 2) else (0, 0, 0, 0)

        if tp == 0:
            warnings.append(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Fold {fold} "
                            f"| Prob Média: {y_pred_prob.mean():.4f} | Threshold: {best_threshold:.4f}")

        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan

        print(f"Fold {fold}/{n_splits} - Accuracy --> {accuracy}...")

        results.append({
            "Fold": fold,
            "y_true": list(y_test),  # 🟢 Adicionando a coluna faltante
            "y_pred": list(y_pred),  # 🟢 Adicionando também y_pred para compatibilidade
            "y_pred_prob": list(y_pred_prob),  # 🟢 Adicionando as probabilidades para análise
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "Accuracy": accuracy,
            "ROC-AUC": roc_auc_score(y_test, y_pred_prob) if len(set(y_test)) > 1 else np.nan,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp,
            "best_threshold": best_threshold
        })

    # Média das métricas de validação ao longo das épocas
    mean_history_log = {
        "epoch": history_log["epoch"],
        "val_loss": [np.mean(epoch_values) for epoch_values in history_log["val_loss"]],
        "val_f1_score": [np.mean(epoch_values) for epoch_values in history_log["val_f1_score"]],
    }
    plot_mean_validation_metrics(mean_history_log)

    if results:
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(RESULTS_PATH, "kfold_results.csv"), index=False)
        print(f"📊 Resultados K-Fold salvos em {RESULTS_PATH}")

        # Exibe métricas médias finais
        print("\n📌 Resumo das métricas (Média entre os Folds):")
        print(results_df.mean(numeric_only=True))

    else:
        print("[ERRO] Nenhum resultado foi gerado para salvar em kfold_results.csv")

    # Se houver warnings, salva-os
    if warnings:
        with open(os.path.join(RESULTS_PATH, "kfold_warnings.log"), "a") as log_file:
            log_file.write("\n".join(warnings) + "\n")

    # Gerar o resumo das métricas finais
    analyze_results(results_df, "kfold")

    return results_df


# Executa teste leave-one-out cross validation
def run_loocv(activation, dropout_rate, learning_rate, n_layers, n_neurons, n_epochs, batch_size, early_stop_patience,
              lr_scheduler_patience, model_type, use_augmentation=False, use_shap=False, sample_size=None):

    warnings = []
    # Criar listas para armazenar os valores de loss e f1_score ao longo das épocas
    history_log = {"epoch": [], "val_loss": [], "val_f1_score": []}

    df = load_data()
    labels = df["category"].replace({'clear': 1, 'non-clear': 0}).to_numpy().astype(int)
    features = feature_model_extract(df, model_type, use_augmentation, use_shap, sample_size)

    loo = LeaveOneOut()
    results = []
    total_samples = len(df)

    warnings_file = os.path.join(RESULTS_PATH, "warnings.log")
    if os.path.exists(warnings_file):
        os.remove(warnings_file)  # Remove arquivo antigo antes de iniciar nova rodada

    for i, (train_idx, test_idx) in enumerate(loo.split(features), 1):
        print(f"Processando imagem {i}/{total_samples}...")
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]
        model = get_classifier_model(X_train.shape[1], activation, dropout_rate, learning_rate, n_layers, n_neurons)

        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stop_patience, min_delta=0.001,
                                       restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=lr_scheduler_patience, min_lr=1e-5)

        # Separar 80% para treino e 20% para validação
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)

        # Converter labels de validação para float32
        y_train = y_train.astype("float32")
        y_val = y_val.astype("float32")

        # treina o modelo
        history = model.fit(X_train, y_train,
                      epochs=n_epochs,
                      batch_size=min(batch_size, len(X_train)),
                      validation_data=(X_val, y_val),
                      callbacks=[early_stopping, lr_scheduler],
                      verbose=0)

        #print("Métricas disponíveis no history:", history.history.keys())

        # 📌 Armazena os valores das métricas ao longo das épocas
        for epoch, (vloss, vf1) in enumerate(zip(history.history["val_loss"], history.history["val_f1_metric"])):
            if epoch >= len(history_log["epoch"]):
                history_log["epoch"].append(epoch)
                history_log["val_loss"].append([])
                history_log["val_f1_score"].append([])

            history_log["val_loss"][epoch].append(vloss)
            history_log["val_f1_score"][epoch].append(vf1)

        # realiza a predição
        y_pred_prob = model.predict(X_test)[0][0]

        # determina o melhor threshold para o conjunto.
        best_threshold = find_best_threshold(y_test, y_pred_prob)

        y_pred = (y_pred_prob > best_threshold).astype("int32")
        cm = confusion_matrix([y_test], [y_pred], labels=[0, 1])
        # Correção para evitar erro se a matriz de confusão não for 2x2
        tn, fp, fn, tp = cm.flatten() if cm.shape == (2, 2) else (0, 0, 0, 0)

        # 🚨 Verificação de TP = 0 (Nenhum Verdadeiro Positivo)
        if tp == 0:
            warnings.append(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - "
                            f"Arquivo: {df.iloc[test_idx]['filename'].values[0]} "
                            f"| Prob: {y_pred_prob:.4f} | Threshold: {best_threshold:.4f}")

        results.append({
            "filename": df.iloc[test_idx]["filename"].values[0],
            "y_true": y_test.item(),
            "y_pred": y_pred,
            "y_pred_prob": y_pred_prob,
            "best_threshold": best_threshold,
            "Precision": precision_score([y_test], [y_pred], zero_division=0),
            "Recall": recall_score([y_test], [y_pred], zero_division=0),
            "F1-Score": f1_score([y_test], [y_pred], zero_division=0),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp
        })

    # Converte os valores armazenados para médias por época
    mean_history_log = {
        "epoch": history_log["epoch"],
        "val_loss": [np.mean(epoch_values) for epoch_values in history_log["val_loss"]],
        "val_f1_score": [np.mean(epoch_values) for epoch_values in history_log["val_f1_score"]],
    }
    plot_mean_validation_metrics(mean_history_log)

    mean_history_log_df = pd.DataFrame([mean_history_log])
    mean_history_log_df.to_csv(os.path.join(RESULTS_PATH, "mean_history_log.csv"), index=False)
    print(f"Resumo mean_history_log salvo em {RESULTS_PATH}")

    print("\nProcessamento concluído!")
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_PATH, "loocv_results.csv"), index=False)
    print(f"Resultados LOOCV salvos em {RESULTS_PATH}")

    if warnings:
        with open(os.path.join(RESULTS_PATH, "warnings.log"), "a") as log_file:
            log_file.write("\n".join(warnings) + "\n")

    analyze_results(results_df)
    plot_threshold_distribution(results_df)

# --- Analise os Resultados - Saídas do modelo para avaliação
def plot_threshold_distribution(df):
    plt.figure(figsize=(8, 6))  # Define um tamanho adequado para o gráfico
    plt.hist(df["best_threshold"], bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(df["best_threshold"].median(), color='red', linestyle='dashed', label='Mediana')
    plt.xlabel("Threshold")
    plt.ylabel("Frequência")
    plt.title("Distribuição dos Thresholds Otimizados")
    plt.legend()

    # Define o caminho onde o gráfico será salvo
    save_path = os.path.join(RESULTS_PATH, "threshold_distribution.png")

    # 🔹 Salva primeiro para garantir que a imagem é escrita no arquivo antes de ser exibida
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Gráfico salvo em: {save_path}")

    # Agora sim, exibe o gráfico
    plt.show()

    # Fecha a figura para liberar memória
    plt.close()

def analyze_results(df, test_type="loocv"):
    median_threshold = calculate_median_threshold(df)

    if test_type == "loocv":
        y_true = df["y_true"]
        y_pred = df["y_pred"]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if len(set(y_true)) > 1:
            roc_auc = roc_auc_score(y_true, df["y_pred_prob"])
        else:
            roc_auc = np.nan
            print("[AVISO] ROC-AUC não pode ser calculado pois y_true contém apenas uma classe (tudo 0 ou tudo 1).")

        summary = {
            "Median Threshold (Filtered)": median_threshold,
            "Accuracy": (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else np.nan,
            "Precision": precision, "Recall": recall, "F1-Score": f1,
            "ROC-AUC": None if np.isnan(roc_auc) else roc_auc,
            "True Negatives": tn, "False Positives": fp, "False Negatives": fn, "True Positives": tp
        }
    else:
        summary = {
            "Median Threshold (Filtered)": median_threshold,
            "Accuracy": df["Accuracy"].mean(),
            "Precision": df["Precision"].mean(),
            "Recall": df["Recall"].mean(),
            "F1-Score": df["F1-Score"].mean(),
            "ROC-AUC": df["ROC-AUC"].mean(skipna=True),
            "True Negatives": df["TN"].mean(),
            "False Positives": df["FP"].mean(),
            "False Negatives": df["FN"].mean(),
            "True Positives": df["TP"].mean()
        }

    summary_df = pd.DataFrame([summary])

    summary_filename = "kfold_summary.csv" if "Fold" in df.columns else "loocv_summary.csv"
    summary_df.to_csv(os.path.join(RESULTS_PATH, summary_filename), index=False)

    print("Resumo Final das Métricas:")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
    print(f"Resumo das métricas salvo em {RESULTS_PATH}")

def plot_mean_validation_metrics(mean_history_log):
    plt.figure(figsize=(10, 6))

    # Plota a média de val_loss e val_f1_score ao longo das épocas
    plt.plot(mean_history_log["epoch"], mean_history_log["val_loss"], label="Média Val Loss", color="blue", linestyle='-')
    plt.plot(mean_history_log["epoch"], mean_history_log["val_f1_score"], label="Média Val F1-Score", color="red", linestyle='--')

    plt.xlabel("Época")
    plt.ylabel("Valor Médio")
    plt.title("Evolução Média de Val Loss e Val F1-Score por Época")
    plt.legend()

    # Salvar e exibir o gráfico
    save_path = os.path.join(RESULTS_PATH, "mean_val_loss_vs_val_f1_score.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"📊 Gráfico salvo em: {save_path}")
    plt.show()

if __name__ == "__main__":
    activation = 'relu'
    dropout_rate = 0.1
    learning_rate = 0.0001 # 0.0005 # 0.001
    n_layers = 1 #2
    n_neurons = 256 #512
    batch_size = 64
    n_epochs = 1000
    early_stop_patience = 150 #100 #50
    lr_scheduler_patience = 10
    model_type = 'MobileNetV1'
    n_splits = 10
    use_augmentation = False
    use_shap = True
    sample_size = 100

    #Para rodar teste de leave-one-out cross validation descomente abaixo
    #run_loocv(activation, dropout_rate, learning_rate, n_layers, n_neurons, n_epochs, batch_size, early_stop_patience,
    #          lr_scheduler_patience, model_type, use_augmentation, use_shap, sample_size)

    #Para rodar teste de kfold cross validation - descomente abaixo
    run_kfold_cv(activation, dropout_rate, learning_rate, n_layers, n_neurons,
                 n_epochs, batch_size, early_stop_patience, lr_scheduler_patience,
                 model_type, n_splits, use_augmentation, use_shap, sample_size)

