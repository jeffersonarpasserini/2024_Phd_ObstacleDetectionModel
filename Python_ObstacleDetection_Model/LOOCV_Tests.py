import os
import numpy as np
import pandas as pd
import tensorflow as tf
import random
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import BinaryCrossentropy
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

model_type = 'MobileNetV1'


def build_model(input_shape, activation, dropout_rate, learning_rate, n_layers, n_neurons):
    """
    Constrói um modelo de rede neural totalmente configurável.

    Args:
        input_shape (int): Dimensão da entrada.
        activation (str): Função de ativação para as camadas ocultas.
        dropout_rate (float): Taxa de dropout (se 0, a camada não é adicionada).
        learning_rate (float): Taxa de aprendizado do otimizador.
        n_layers (int): Número de camadas ocultas.
        n_neurons (int): Número de neurônios por camada oculta.

    Returns:
        model: Modelo compilado do TensorFlow/Keras.
    """
    model = Sequential()
    model.add(Input(shape=(input_shape,)))

    for _ in range(n_layers):
        model.add(Dense(n_neurons, activation=activation))
        if dropout_rate > 0:
            model.add(Dropout(dropout_rate))

    model.add(Dense(1, activation='sigmoid'))

    optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=BinaryCrossentropy(), metrics=['accuracy'])
    return model


def run_loocv(activation, dropout_rate, learning_rate, n_layers, n_neurons):
    df = load_data()
    labels = df["category"].replace({'clear': 1, 'non-clear': 0}).to_numpy().astype(int)
    features = modular_extract_features(df, model_type)

    loo = LeaveOneOut()
    results = []
    total_samples = len(df)

    for i, (train_idx, test_idx) in enumerate(loo.split(features), 1):
        print(f"Processando imagem {i}/{total_samples}...")

        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        model = build_model(X_train.shape[1], activation, dropout_rate, learning_rate, n_layers, n_neurons)
        model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=0)

        y_pred_prob = model.predict(X_test)[0][0]
        y_pred = 1 if y_pred_prob > 0.5 else 0  # Garantir predição correta

        cm = confusion_matrix([y_test], [y_pred])
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

        results.append({
            "filename": df.iloc[test_idx]["filename"].values[0],
            "y_true": y_test.item(),
            "y_pred": y_pred,
            "y_pred_prob": y_pred_prob,
            "Precision": precision_score([y_test], [y_pred], zero_division=0),
            "Recall": recall_score([y_test], [y_pred], zero_division=0),
            "F1-Score": f1_score([y_test], [y_pred], zero_division=0),
            "TN": tn, "FP": fp, "FN": fn, "TP": tp
        })

    print("\nProcessamento concluído!")
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_PATH, "loocv_results.csv"), index=False)
    print(f"Resultados LOOCV salvos em {RESULTS_PATH}")

    analyze_results(results_df)


def analyze_results(df):
    """
    Calcula a matriz de confusão e gera métricas globais para o modelo.
    """
    """
    Calcula a matriz de confusão e gera métricas globais para o modelo.
    """
    y_true = df["y_true"]
    y_pred = df["y_pred"]
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    # Calcular métricas globais
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, df["y_pred_prob"]) if len(set(y_true)) > 1 else np.nan
    scores = df["y_pred_prob"]
    Q1, Q2, Q3 = scores.quantile([0.25, 0.5, 0.75])
    IQR = Q3 - Q1
    lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    filtered_scores = scores[(scores >= lower_bound) & (scores <= upper_bound)]
    median_no_outliers = filtered_scores.median()

    # Calcular ROC-AUC após processar todas as imagens
    if len(set(df["y_true"])) > 1:
        roc_auc = roc_auc_score(df["y_true"], df["y_pred_prob"])
    else:
        roc_auc = np.nan

    summary = {
        "Accuracy": (tp + tn) / (tp + tn + fp + fn),
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "True Negatives": tn,
        "False Positives": fp,
        "False Negatives": fn,
        "True Positives": tp
    }

    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(os.path.join(RESULTS_PATH, "loocv_summary.csv"), index=False)

    print("Resumo Final das Métricas:")
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")

    print(f"Resumo das métricas salvo em {RESULTS_PATH}")


if __name__ == "__main__":
    activation = 'relu'
    dropout_rate = 0
    learning_rate = 0.0005
    n_layers = 2
    n_neurons = 256

    run_loocv(activation, dropout_rate, learning_rate, n_layers, n_neurons)


