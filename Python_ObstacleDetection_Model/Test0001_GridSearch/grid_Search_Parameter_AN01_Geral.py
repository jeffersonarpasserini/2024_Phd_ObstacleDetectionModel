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

# Dividir os dados em treino e validação
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=SEED)
print('Dataset splitted...')

# Treinar o modelo com as features extraídas
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5)

# ✅ Função para criar o modelo
def get_classifier_model(learning_rate=0.01, activation='relu', n_layers=1, n_neurons=64, dropout_rate=0.0):
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

# ✅ Definir a grade de hiperparâmetros
param_grid = {
    'learning_rate': [0.0005, 0.0001, 0.005, 0.001, 0.01, 0.1],
    'activation': ['relu', 'tanh'],
    'n_layers': [1, 2, 3],
    'n_neurons': [16, 32, 64, 128, 256, 512],
    'dropout_rate': [0.0, 0.2, 0.5]
}

# ✅ Criar modelo para GridSearchCV com `model=`
model = KerasClassifier(
    build_fn=get_classifier_model,
    input_shape=X_train.shape[1],
    verbose=0,
    callbacks=[lr_scheduler]
)

model = KerasClassifier(
    model=create_model,
    learning_rate=0.01,
    activation='relu',
    n_layers=1,
    n_neurons=64,
    dropout_rate=0.0,
    verbose=0,
)

# ✅ Configurar GridSearchCV com validação cruzada estratificada
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
grid_search = GridSearchCV(
    estimator=model,
    param_grid=params_grid,
    scoring='accuracy',
    cv=StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED),
    refit=True,
    n_jobs=1,
    verbose=0,
    error_score='raise'
)

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=cv,
    refit=True,  # Mantém o melhor modelo treinado
    n_jobs=1,  # Evita erros de paralelismo
    verbose=2
)

# ✅ Executar a busca
grid_result = grid_search.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, lr_scheduler]
)

# ✅ Criar DataFrame com os resultados do GridSearchCV
results_df = pd.DataFrame(grid_result.cv_results_)

# ✅ Salvar os resultados em um arquivo CSV
results_df.to_csv("grid_search_results.csv", index=False)

print("Resultados do GridSearchCV salvos em 'grid_search_results.csv'")

# ✅ Exibir os melhores hiperparâmetros encontrados
print("Melhores hiperparâmetros: ", grid_search.best_params_)
print("Melhor acurácia: ", grid_search.best_score_)
