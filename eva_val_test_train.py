import torch

def validate(model, device, valid_loader, criterion):
    model.eval()
    valid_loss = 0
    try:
        with torch.no_grad():
            for data, target in valid_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                valid_loss += criterion(output, target).item()  # sum up batch loss
        valid_loss /= len(valid_loader.dataset)
        print('Validation set: Average loss: {:.4f}\n'.format(valid_loss))

    except Exception as e:
        print('Exception occurred: ' + str(e))

    return valid_loss


def evaluate(model, val_loader, criterion, device):
    val_loss = []
    model.eval()
    with torch.no_grad():
        for data, target in val_loader:
            try:
                data, target = data.to(device), target.to(device)
                output = model(data)

                loss = criterion(output, target.argmax(dim=1))
                val_loss.append(loss.item())
            except Exception as e:
                print(f'Exception occurred: {str(e)}')
                continue

    return np.mean(val_loss)


def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            try:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += criterion(output, target).item()
            except Exception as e:
                print(f'Exception occurred: {str(e)}')
                continue
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}\n'.format(test_loss))
    return test_loss


def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        try:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if batch_idx % 10 == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))
        except Exception as e:
            print(f'Exception occurred in batch {batch_idx}: {str(e)}')
    train_loss /= len(train_loader.dataset)

    # Save the model after each epoch
    save_path = f"savedd_models/model_epoch_{epoch}.pt"
    torch.save(model.state_dict(), save_path)

    return train_loss