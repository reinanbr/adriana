
from adrianna.neuro.base_v1 import NeuralNetwork
import numpy as np


# Exemplo de uso
if __name__ == "__main__":
    # Dados de entrada e saída
    X = np.array([[3, 1, 2], [1, 24, 5], [2, 42, 5], [2, 23, 3]])
    y = np.array([[1], [0], [1], [0]])
    
    # Criação e treinamento da rede neural
    input_size = X.shape[1]  # Número de colunas das listas
    hidden_size = 4  # Número de neurônios na camada oculta
    output_size = y.shape[1]  # Número de saídas
    
    neural_net = NeuralNetwork(input_size, hidden_size, output_size, learning_rate=0.1)
    neural_net.train(X, y, epochs=10000)
    
    # Fazendo previsões
    predictions = neural_net.predict(X)
    predictions = np.round(predictions).astype(int)
    print("Predictions:")
    print(predictions)
