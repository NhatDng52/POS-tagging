import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)
class CustomRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(CustomRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        output, _ = self.rnn(x)
        logits = self.fc(output)
        return logits

    def train_model(self, X, Y, batch_size=128, epochs=200, lr=0.02):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        X_tensor = X.clone().detach().float().to(device)
        Y_tensor = Y.clone().detach().float().to(device)


        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.parameters(), lr=lr)

        dataset = torch.utils.data.TensorDataset(X_tensor, Y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.train()
            total_loss = 0
            for x_batch, y_batch in loader:
                optimizer.zero_grad()
                output = self(x_batch)  # (batch, seq_len, output_dim)
                output = output.view(-1, output.shape[-1])
                y_batch = y_batch.view(-1, y_batch.shape[-1])

                loss = criterion(output, torch.argmax(y_batch, dim=1))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            # print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")