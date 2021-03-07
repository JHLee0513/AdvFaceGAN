from loss import combined_loss
import torch.nn as nn
import torch.optim as optim


"""
Training script for AdvFaceGAN
"""

EPOCHS = 10
BATCHSIZE = 8
INPUT_SIZE = 112
TARGET = 4


def step(model, batch, criterion):
    predictions = model(batch)
    loss = criterion(predictions)
    return predictions, loss

def validate(model, valid_loader, criterion):
    valid_loss = []
    model.eval()
    for data in valid_loader:
        pred, loss = step(model, data, criterion)
        valid_loss += [loss]

    return np.array(valid_loss).mean()

def train(model, train_loader, criterion, optimizer):
    train_loss = []
    model.train()

    for data in train_loader: # <-- emore dataset dataloader
        optimizer.zero_grad()
        pred, loss = step(model, data, criterion)
        # get prediction of the facial recognition model of G(x) + x
        train_loss += [loss]
        loss.backward()
        optimizer.step()

    return np.asarray(train_loss).mean()

def main():
    train_dataset = None
    train_loader = None
    valid_dataset = None
    valid_loader = None
    model = None
    criterion = combined_loss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS, min=0)

    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, criterion)
        valid_loss = validate(model, valid_loader, criterion)
        
        scheduler.step()

if __name__ == "__main__":
    # setup network and modules
    # define parameters
    # run training & validation loop
    main()