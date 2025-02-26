import os
import numpy as np
import tensorflow as tf
import pandas as pd  # Biblioteca para salvar os resultados
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import extract_features

# Configuração de ambiente determinístico
os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Carregar os dados
df = extract_features.load_data()
model_type = 'MobileNetV1'

# Extrair features
features = extract_features.modular_extract_features(df, model_type)
print('Features extracted...')

# Criar labels binárias de forma segura
labels = df["category"].map({'clear': 1, 'non-clear': 0}).astype(int)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=SEED)
print('Dataset splitted...')

# Definir callbacks corretamente
early_stopping = EarlyStopping(monitor='val_loss', patience=50, min_delta=0.001, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-5)


# Função para criar o modelo
def create_model(learning_rate=0.01, activation='relu', n_layers=1, n_neurons=64, dropout_rate=0.0):
    model = Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
    model.add(Dense(n_neurons, activation=activation))
    for _ in range(n_layers - 1):
        model.add(Dense(n_neurons, activation=activation))
        if dropout_rate > 0.0:
            model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    optimizer = RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
    return model


# Definir a grade de hiperparâmetros
# param_grid = {
#   'model__learning_rate': [0.0005, 0.0001, 0.005, 0.001, 0.01, 0.1],
#    'model__activation': ['relu', 'tanh'],
#    'model__n_layers': [1, 2, 3],
#    'model__n_neurons': [16, 32, 64, 128, 256, 512],
#    'model__dropout_rate': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
#}

param_grid = {
    'model__learning_rate': [0.001],
    'model__activation': ['relu'],
    'model__n_layers': [2],
    'model__n_neurons': [512],
    'model__dropout_rate': [0.1]
}

# Criar modelo para GridSearchCV incluindo callbacks corretamente
model = KerasClassifier(
    model=create_model,
    verbose=0,
    callbacks=[lr_scheduler]
)

# Configurar GridSearchCV com os hiperparâmetros corrigidos
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED),
    refit=True,  # Mantém o melhor modelo treinado
    n_jobs=1,
    verbose=0
)

# Executar a busca com validação cruzada
grid_result = grid_search.fit(
    X_train, y_train,
    validation_split=0.2,
    callbacks=[lr_scheduler]
)

# Criar DataFrame com os resultados do GridSearchCV
results_df = pd.DataFrame(grid_result.cv_results_)
results_df.to_csv("grid_search_results.csv", index=False)
print("Resultados do GridSearchCV salvos em 'grid_search_results.csv'")

# Exibir os melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros: ", grid_search.best_params_)
print("Melhor acurácia: ", grid_search.best_score_)

# Avaliação de todos os modelos treinados com X_test
test_results = []
individual_classifications = []

for idx, params in enumerate(grid_result.cv_results_['params']):
    print(f"🔍 Testando modelo {idx + 1} com parâmetros: {params}")

    # Remover o prefixo 'model__' dos parâmetros antes de passar para a função
    clean_params = {key.replace("model__", ""): value for key, value in params.items()}

    model = create_model(**clean_params)
    model.fit(X_train, y_train, epochs=1000, callbacks=[early_stopping, lr_scheduler], verbose=0)

    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)


    print("------------------------Modelo:", idx+1, "Parâmetros:", clean_params, " Accuracy: ", acc)

    test_results.append({
        "Modelo": idx + 1,
        "Parâmetros": clean_params,
        "Acurácia": acc,
        "Precisão": precision,
        "Recall": recall,
        "F1-Score": f1
    })

    for i, pred in enumerate(y_pred):
        individual_classifications.append({
            "Modelo": idx + 1,
            "Imagem": i,
            "Real": y_test.iloc[i],
            "Predito": pred[0]
        })

# Salvar os resultados em arquivos CSV
test_results_df = pd.DataFrame(test_results)
test_results_df.to_csv("test_results_summary.csv", index=False)

individual_classifications_df = pd.DataFrame(individual_classifications)
individual_classifications_df.to_csv("test_results_individual.csv", index=False)

print("📁 Resultados resumidos salvos em 'test_results_summary.csv'")
print("📁 Resultados individuais salvos em 'test_results_individual.csv'")