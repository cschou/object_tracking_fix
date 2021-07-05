import torch
from torch.utils.data import DataLoader
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import math
from tqdm import tqdm
from model import BLSTM
from datasets import CoordinatesDataset
from loss import EuclideanDist

def evaluate(model, loader):
    model.eval()
    with torch.no_grad():
        loss_total = 0.
        count = 0
        print('evaluate at the epoch end...')
        for batch_idx, (data, targets) in enumerate(loader):
            data = data.to(device)
            targets = targets.to(device)

            predictions = model(data)
            loss = loss_fn(predictions, targets)
            loss_total += loss
            count += 1
        
    return loss_total / count




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Hyperparameters
input_size = 2
sequence_length = 100
num_layers = 2
hidden_size = 256
num_classes = 2
learning_rate = 0.001
batch_size = 64
num_epochs = 10
x_file = 'data/train/x_train.npy'
y_file = 'data/train/y_train.npy'
x_val = 'data/val/x_val.npy'
y_val = 'data/val/y_val.npy'

model = BLSTM(input_size, hidden_size, num_layers, num_classes).to(device)
train_dataset = CoordinatesDataset(x_file, y_file)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = CoordinatesDataset(x_val, y_val)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
loss_fn = EuclideanDist()
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)
n_iter = len(train_dataset) // batch_size
# Train Network
min_val_loss = math.inf
patience = 3
for epoch in range(num_epochs):
    print(f'Processing epoch {epoch}')
    pbar = tqdm(total=n_iter)
    model.train()
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device=device)
        targets = targets.to(device=device)

        # forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        loss.backward()

        # gradient descent or adam step
        optimizer.step()
        if batch_idx % 10 == 0:
            pbar.set_description('train loss: {:.2f}'.format(loss))
            pbar.refresh()
        pbar.update(1)
    epoch_val_loss = evaluate(model, val_loader)
    if epoch_val_loss < min_val_loss:
        print(f'**val loss drops from {min_val_loss} to {epoch_val_loss}, saving model to model.pt')
        min_val_loss = epoch_val_loss
        torch.save(model.state_dict(), 'model.pt')
    else:
        patience -= 1
        if patience == 0:
            lr_scheduler.step()
            print('**reducing learning rate to 1/10')
            patience = 3

    print(f'eval loss: {epoch_val_loss:.3f}')
    pbar.close()
