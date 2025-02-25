import datetime
import os
import numpy as np
import pandas as pd
import scipy
import tensorflow as tf
import random

from matplotlib import pyplot as plt
from scipy.optimize import minimize_scalar
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
from extract_features import load_data, modular_extract_features

# Configuração de semente para reprodutibilidade
SEED = 1980
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Definir caminhos
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(BASE_PATH, 'results_details', 'loocv_results')
os.makedirs(RESULTS_PATH, exist_ok=True)


def build_model(input_shape, activation, dropout_rate, learning_rate, n_layers, n_neurons):
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

# executa teste de validação cruzada
def run_kfold_cv(activation, dropout_rate, learning_rate, n_layers, n_neurons,
                 n_epochs, batch_size, early_stop_patience, lr_scheduler_patience,
                 model_type, n_splits=5):
    warnings = []
    history_log = {"epoch": [], "val_loss": [], "val_f1_score": []}
    df = load_data()

    labels = df["category"].replace({'clear': 1, 'non-clear': 0}).to_numpy().astype(int)
    features = modular_extract_features(df, model_type)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(features, labels), 1):
        print(f"📌 Rodando Fold {fold}/{n_splits}...")

        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        model = build_model(X_train.shape[1], activation, dropout_rate, learning_rate, n_layers, n_neurons)

        early_stopping = EarlyStopping(monitor='val_loss', patience=early_stop_patience, min_delta=0.001,
                                       restore_best_weights=True)
        lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=lr_scheduler_patience, min_lr=1e-5)

        # Divide os dados de treino para criar um conjunto de validação (80% treino / 20% validação)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=SEED)

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

        results.append({
            "Fold": fold,
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, zero_division=0),
            "ROC-AUC": roc_auc_score(y_test, y_pred_prob) if len(set(y_test)) > 1 else np.nan,
            "TN": tn, "FP": fp, "FN": fn, "TP": tp
        })

    # Média das métricas de validação ao longo das épocas
    mean_history_log = {
        "epoch": history_log["epoch"],
        "val_loss": [np.mean(epoch_values) for epoch_values in history_log["val_loss"]],
        "val_f1_score": [np.mean(epoch_values) for epoch_values in history_log["val_f1_score"]],
    }
    plot_mean_validation_metrics(mean_history_log)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_PATH, "kfold_results.csv"), index=False)
    print(f"📊 Resultados K-Fold salvos em {RESULTS_PATH}")

    # Exibe métricas médias finais
    print("\n📌 Resumo das métricas (Média entre os Folds):")
    print(results_df.mean(numeric_only=True))

    # Se houver warnings, salva-os
    if warnings:
        with open(os.path.join(RESULTS_PATH, "kfold_warnings.log"), "a") as log_file:
            log_file.write("\n".join(warnings) + "\n")

    return results_df

# executa teste leave-one-out cross validation
def run_loocv(activation, dropout_rate, learning_rate, n_layers, n_neurons, n_epochs, batch_size, early_stop_patience,
              lr_scheduler_patience, model_type):

    warnings = []
    # Criar listas para armazenar os valores de loss e f1_score ao longo das épocas
    history_log = {"epoch": [], "val_loss": [], "val_f1_score": []}

    df = load_data()
    labels = df["category"].replace({'clear': 1, 'non-clear': 0}).to_numpy().astype(int)
    features = modular_extract_features(df, model_type)
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
        model = build_model(X_train.shape[1], activation, dropout_rate, learning_rate, n_layers, n_neurons)

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

def find_best_threshold(y_test, y_pred_prob):
    def neg_f1(thresh):
        y_pred = (y_pred_prob > thresh).astype("int32")
        return -f1_score([y_test], [y_pred], zero_division=0)


    #result = minimize_scalar(neg_f1, bounds=(0.2, 0.6), method='bounded')
    # 🔹 Alteramos o intervalo para evitar thresholds baixos demais e
    # permitimos um threshold um pouco mais alto
    result = minimize_scalar(neg_f1, bounds=(0.3, 0.6), method='bounded')

    return result.x if result.success else 0.6  # Threshold padrão: 0.5 --> ajustado para 0.6 testes anteriores com bons resultados

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

def analyze_results(df):
    median_threshold = calculate_median_threshold(df)
    y_true = df["y_true"]
    y_pred = df["y_pred"]
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
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

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(RESULTS_PATH, "loocv_summary.csv"), index=False)
    print("Resumo Final das Métricas:")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")
    print(f"Resumo das métricas salvo em {RESULTS_PATH}")


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

    #Para rodar teste de leave-one-out cross validation descomente abaixo
    run_loocv(activation, dropout_rate, learning_rate, n_layers, n_neurons, n_epochs, batch_size, early_stop_patience,
              lr_scheduler_patience, model_type)

    #Para rodar teste de validação cruzada - descomente abaixo
    # run_kfold_cv(activation, dropout_rate, learning_rate, n_layers, n_neurons, n_epochs, batch_size, early_stop_patience,
    #             lr_scheduler_patience, model_type, n_splits)
