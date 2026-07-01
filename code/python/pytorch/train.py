import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW


class Net(nn.Module):
    def __init__(self, d_model:int, num_classes:int=5):
        super().__init__()
        self.linear = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x:torch.Tensor):
        x = self.linear(x)
        x = self.norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


dataset = MyDataset(data, labels)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_set, batch_size=32, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_set, batch_size=32, shuffle=False, num_workers=4, pin_memory=True
)
device = "cpu"
num_epochs=5
model = Net(128,5).to(device)
optimizer = AdamW(model.parameters(), lr=3e-4)
criterion = nn.CrossEntropyLoss()


for epoch in range(num_epochs):
    # ── Training ──────────────────────────────────────────────────────
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

    # ── Validation ────────────────────────────────────────────────────
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            val_loss += criterion(model(inputs), targets).item()
    print(f"Epoch {epoch + 1}: val={val_loss / len(val_loader):.4f}")


if __name__=="__main__":

    model = Net(128, 5)
    print(model)

    for name, p in model.named_parameters():
        print(f"{name:40} {tuple(p.shape)}  {p.numel():,}")

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total: {total:,} | Trainable: {trainable:,}")

