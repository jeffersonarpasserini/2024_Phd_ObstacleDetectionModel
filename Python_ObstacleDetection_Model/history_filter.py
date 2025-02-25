import numpy as np

# implementa um filtro de suavização - filtro histórico (sliding window)
# para amenizar as últimas N previsões e usar uma média movel para decidir se há realmente um obstáculo.

#LEMBRAR DE  IMPLEMENTAR NO APLICATIVO MOBILE --> NAO INTERFERE NO TREINAMENTO.

class HistoryFilter:
    def __init__(self, window_size=3):
        """ Inicializa o filtro de suavização com um histórico de tamanho definido. """
        self.history = []
        self.window_size = window_size

    def update(self, new_value):
        """ Adiciona um novo valor à janela deslizante e retorna a média das previsões recentes. """
        self.history.append(new_value)
        if len(self.history) > self.window_size:
            self.history.pop(0)  # Mantém apenas os últimos N valores

        return np.mean(self.history)  # Retorna a média móvel
