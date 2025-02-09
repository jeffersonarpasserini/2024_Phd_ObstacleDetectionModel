import os
import numpy as np
import tensorflow as tf
import pandas as pd  # Biblioteca para salvar os resultados
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, StratifiedKFold
import extract_features

# Configuração de ambiente determinístico
os.environ['TF_DETERMINISTIC_OPS'] = '1'
SEED = 1980
os.environ['PYTHONHASHSEED'] = str(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)

# Carregar os dados
df = extract_features.load_data()

# Extrair features
features = extract_features.modular_extract_features(df)
print('Features extracted...')

# Criar labels binárias
labels = df["category"].replace({'clear': 1, 'non-clear': 0}).astype(int)

# Dividir os dados em treino e validação
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=SEED)
print('Dataset splitted...')

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
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])  # Mantendo apenas acurácia
    return model

# Definir a grade de hiperparâmetros
param_grid = {
    'learning_rate': [0.0005, 0.0001, 0.005, 0.001, 0.01, 0.1],
    'activation': ['relu', 'tanh'],
    'n_layers': [1, 2, 3],
    'n_neurons': [32, 64, 128, 256, 512],
    'dropout_rate': [0.0, 0.2, 0.5]
}

# Criar modelo para GridSearchCV
model = KerasClassifier(
    model=create_model,  # Função para criar o modelo
    learning_rate=0.01,  # Valor padrão
    activation='relu',   # Valor padrão
    n_layers=1,          # Valor padrão
    n_neurons=64,        # Valor padrão
    dropout_rate=0.0,    # Valor padrão
    verbose=0
)

# Configurar GridSearchCV com validação cruzada estratificada
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',  # Apenas acurácia
    cv=cv,
    refit=False,
    n_jobs=-1,
    verbose=2
)

# Executar a busca
grid_result = grid_search.fit(X_train, y_train)

# Criar DataFrame com os resultados do GridSearchCV
results_df = pd.DataFrame(grid_result.cv_results_)

# Salvar os resultados em um arquivo CSV
results_df.to_csv("grid_search_results.csv", index=False)

print("Resultados do GridSearchCV salvos em 'grid_search_results.csv'")

# Exibir os melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros: ", grid_result.best_params_)
print("Melhor acurácia: ", grid_result.best_score_)
