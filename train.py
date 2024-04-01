import torch
from tqdm import tqdm

def GetCorrectPredCount(pPrediction, pLabels):
    """Method to get count of correct predictions

    Args:
        pPrediction (Tensor): Tensor containing prediction from model
        pLabels (Tensor): Tensor containing true labels from data

    Returns:
        int: Number of correct predictions
    """
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

def _train(model, device, train_loader, optimizer, scheduler, criterion, train_losses, train_acc):
    """Method to train model for one epoch 

    Args:
        model (Object): Neural Network model
        device : torch.device indicating available device for training cuda/cpu
        train_loader (Object): Object of DataLoader class for training data
        optimizer (Object): Object of optimizer to update weights
        criterion (Object): To calculate loss
        train_losses (List): To store training loss
        train_acc (List): To store training accuracy
    """
    # Set model to training
    model.train()
    # Create progress bar
    pbar = tqdm(train_loader)

    train_loss = 0
    correct = 0
    processed = 0

    # Loop through batches of data
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device) # Store data and target to device
        optimizer.zero_grad() # Set gradients to zero

        # Predict
        pred = model(data)

        # Calculate loss
        loss = criterion(pred, target)
        train_loss+=loss.item() # Update train loss

        # Backpropagation
        loss.backward() # Compute gradients
        optimizer.step() # Updates weights
        scheduler.step() # Update learning rate

        correct += GetCorrectPredCount(pred, target) # Store correct prediction count
        processed += len(data) # Store amount of data processed

        # Print results
        pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

    # Append accuracy and losses to list
    train_acc.append(100*correct/processed)
    train_losses.append(train_loss/len(train_loader))

def _test(model, device, test_loader, criterion, test_losses, test_acc):
    """Method to test model

    Args:
        model (Object): Neural Network model
        device : torch.device indicating available device for testing cuda/cpu
        test_loader (Object): Object of DataLoader class for testing data
        criterion (Object): To calculate loss
        test_losses (List): To store testing loss
        test_acc (List): To store testing accuracy
    """
    # Set model to eval
    model.eval()

    test_loss = 0
    correct = 0

    # Disable gradient calculation
    with torch.no_grad():
        # iterate though data and calculate loss
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device) # Store data and target to device

            output = model(data) # get prediction
            test_loss += criterion(output, target, reduction='sum').item()  # sum up batch loss

            correct += GetCorrectPredCount(output, target) # Store correct prediction count

    # Append accuracy and losses to list
    test_loss /= len(test_loader.dataset)
    test_acc.append(100. * correct / len(test_loader.dataset))
    test_losses.append(test_loss)

    # Print results
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
def start_training(num_epochs, model, device, train_loader, test_loader, optimizer, criterion, scheduler):
    """Method to start training the model

    Args:
        num_epochs (int): number of epochs
        model (Object): Neural Network model
        device : torch.device indicating available device for training cuda/cpu
        train_loader (Object): Object of DataLoader class for training data
        test_loader (Object): Object of DataLoader class for testing data
        optimizer (Object): Object of optimizer to update weights
        criterion (Object): To calculate loss
        scheduler (Object): To update learning rate

    Returns:
        lists: Lists containing information about losses and accuracy during training and testing
    """
    # Data to plot accuracy and loss graphs
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    for epoch in range(1, num_epochs+1):
        print(f'Epoch {epoch}')
        # Print learning rate
        print(f'Current learning rate: {scheduler.get_last_lr()}')
        # Train for one epochs
        _train(model, device, train_loader, optimizer, scheduler, criterion, train_losses, train_acc)
        # Test model
        _test(model, device, test_loader, criterion, test_losses, test_acc)

    return train_losses, train_acc, test_losses, test_acc