import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Inicialização dos pesos e bias
        self.W1 = np.random.randn(self.input_size, self.hidden_size)
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = np.random.randn(self.hidden_size, self.output_size)
        self.b2 = np.zeros((1, self.output_size))
        
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return (self.sigmoid(x)**2)*np.exp(-x)
    
    def forward(self, X):
        # Camada oculta
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.sigmoid(self.z1)
        
        # Camada de saída
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        # Gradiente da função de perda em relação à saída da camada de saída
        output_error = 2 * (output - y) * self.sigmoid_derivative(output)
        
        # Gradiente da função de perda em relação aos pesos e bias da camada de saída
        dW2 = np.dot(self.a1.T, output_error)
        db2 = np.sum(output_error, axis=0, keepdims=True)
        
        # Gradiente da função de perda em relação à saída da camada oculta
        hidden_error = np.dot(output_error, self.W2.T) * self.sigmoid_derivative(self.a1)
        
        # Gradiente da função de perda em relação aos pesos e bias da camada oculta
        dW1 = np.dot(X.T, hidden_error)
        db1 = np.sum(hidden_error, axis=0)
        
        # Atualização dos pesos e bias
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
        
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                loss = np.mean(np.square((y) - output))
                print(f"Epoch {epoch}, Loss: {loss:.7f}")
               
    def predict(self, X):
        return self.forward(X)
