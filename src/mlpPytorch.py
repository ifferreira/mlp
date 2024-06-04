import torch
import torch.nn as nn
import torch.optim as optim

# Definindo o modelo MLP
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        hidden = self.sigmoid(self.fc1(x))
        output = self.sigmoid(self.fc2(hidden))
        return output

# Dados de treinamento XOR
train_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
target = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# Parâmetros do modelo
input_size = 2
hidden_size = 2
output_size = 1

# Inicializando o modelo
model = MLP(input_size, hidden_size, output_size)

# Função de perda e otimizador
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Número de épocas de treinamento
num_epochs = 50000

for epoch in range(num_epochs):
    # Forward pass
    outputs = model(train_data)
    loss = criterion(outputs, target)

    # Backward pass e otimização
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Imprimindo a perda a cada 1000 épocas
    if (epoch+1) % 1000 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Testando o modelo
with torch.no_grad():
    for datapoint in train_data:
        output = model(datapoint)
        predicted_class = 1 if output >= 0.5 else 0
        print(f'Entrada: {datapoint.numpy()}, Saída Prevista: {output.item()}, Classe Prevista: {predicted_class}')