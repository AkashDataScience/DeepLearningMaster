import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch_lr_finder import LRFinder
from DeepLearningMaster.Datasets import dataset

CLASS_NAMES= ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def plot_samples(train_loader):
    """Method to plot samples of augmented images

    Args:
        train_loader (Object): Object of data loader class to get images
    """
    inv_transform = dataset.get_inv_transforms()

    figure = plt.figure(figsize=(20,20))
    num_of_images = 10
    images, labels = next(iter(train_loader))

    for index in range(1, num_of_images + 1):
        plt.subplot(5, 5, index)
        plt.title(CLASS_NAMES[labels[index].numpy()])
        plt.axis('off')
        image = np.array(images[index])
        image = np.transpose(image, (1, 2, 0))
        image = inv_transform(image=image)['image']
        plt.imshow(image)

def get_SGD_optimizer(model, lr, momentum):
    """Method to get object of stochastic gradient descent. Used to update weights.

    Args:
        model (Object): Neural Network model
        lr (float): Value of learning rate
        momentum (float): Value of momentum

    Returns:
        object: Object of SGD class to update weights
    """
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    return optimizer

def get_Adam_optimizer(model, lr, weight_decay):
    """Method to get object of adam optimizer. Used to update weights.

    Args:
        model (Object): Neural Network model
        lr (float): Value of learning rate
        
    Returns:
        object: Object of SGD class to update weights
    """
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return optimizer

def get_StepLR_scheduler(optimizer, step_size, gamma):
    """Method to get object of scheduler class. Used to update learning rate

    Args:
        optimizer (Object): Object of optimizer
        step_size (int): Period of learning rate decay
        gamma (float): Number to multiply with learning rate

    Returns:
        object: Object of StepLR class to update learning rate
    """
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, verbose=True)
    return scheduler

def get_ReduceLROnPlateau_scheduler(optimizer, factor, patience):
    """Method to get object of scheduler class. Used to update learning rate

    Args:
        optimizer (Object): Object of optimizer
        factor (float): Number to multiply with learning rate
        patience (int): Number of epoch to wait

    Returns:
        object: Object of StepLR class to update learning rate
    """
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=factor, patience=patience, verbose=True)
    return scheduler

def get_OneCycleLR_scheduler(optimizer, max_lr, epochs, steps_per_epoch, max_at_epoch, anneal_strategy, div_factor, final_div_factor):
    """Method to get object of scheduler class. Used to update learning rate

    Args:
        optimizer (Object): Object of optimizer
        max_lr (float): Maximum learning rate to reach during training
        epochs (float): Total number of epoch
        steps_per_epoch (int): Total steps in an epoch
        max_at_epoch (int): Epoch to reach maximum learning rate
        anneal_strategy (string): Strategy to interpolate between minimum and maximum lr
        div_factor (int): Divisive factor to calculate intial learning rate
        final_div_factor (int): Divisive factor to calculate minimum learning rate

    Returns:
        object: Object of StepLR class to update learning rate
    """
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lr,  epochs=epochs,
                                              steps_per_epoch=steps_per_epoch, 
                                              pct_start=max_at_epoch/epochs,
                                              anneal_strategy=anneal_strategy,
                                              div_factor=div_factor,
                                              final_div_factor=final_div_factor)
    return scheduler

def get_nll_loss():
    """Method to get loss calculation ctiterion

    Returns:
        object: Object to calculate loss 
    """
    criterion = F.nll_loss
    return criterion

def get_cross_entropy_loss():
    """Method to get loss calculation ctiterion

    Returns:
        object: Object to calculate loss 
    """
    criterion = F.cross_entropy
    return criterion

def get_learning_rate(model, optimizer, criterion, device, trainloader):
    """Method to find learning rate using LR finder.

    Args:
        model (Object): Object of model
        optimizer (Object): Object of optimizer class
        criterion (Object): Loss function
        device (string): Type of device "cuda" or "cpu"
        trainloader (Object): Object of dataloader class

    Returns:
        float: Learning rate suggested by lr finder
    """
    # Create object and perform range test
    lr_finder = LRFinder(model, optimizer, criterion, device=device)
    lr_finder.range_test(trainloader, end_lr=100, num_iter=100)

    # Plot result and store suggested lr
    plot, suggested_lr = lr_finder.plot()

    # Reset model and optimizer
    lr_finder.reset()
    return suggested_lr

def plot_accuracy_loss_graphs(train_losses, train_acc, test_losses, test_acc):
    """Method to plot loss and accuracy of training and testing

    Args:
        train_losses (List): List containing loss of model after each epoch on training data
        train_acc (List): List containing accuracy of model after each epoch on training data
        test_losses (List): List containing loss of model after each epoch on testing data
        test_acc (List): List containing accuracy of model after each epoch on testing data
    """
    fig, axs = plt.subplots(2,2,figsize=(15,10))
    # Plot training losses
    axs[0, 0].plot(train_losses)
    axs[0, 0].set_title("Training Loss")
    # Plot training accuracy
    axs[1, 0].plot(train_acc)
    axs[1, 0].set_title("Training Accuracy")
    # Plot test losses
    axs[0, 1].plot(test_losses)
    axs[0, 1].set_title("Test Loss")
    # Plot test aacuracy
    axs[1, 1].plot(test_acc)
    axs[1, 1].set_title("Test Accuracy")

def plot_missclassified_images(device, model, test_loader):
    """Method to plot missclassified images

    Args:
        device (string): Type of device "cuda" or "cpu"
        model (Object): Object of model
        test_loader (Object): Object of dataloader class
    """
    model.eval()
    inv_transform = dataset.get_inv_transforms()
    missclassified_image_list = []
    label_list = []
    pred_list = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            if len(missclassified_image_list) > 10:
                break
            for i in range(len(pred)):
                if pred[i] != target[i]:
                    missclassified_image_list.append(data[i])
                    label_list.append(CLASS_NAMES[target[i]])
                    pred_list.append(CLASS_NAMES[pred[i]])

    figure = plt.figure(figsize=(20,20))
    num_of_images = 10
    for index in range(1, num_of_images + 1):
        plt.subplot(5, 5, index)
        plt.title(f'Actual: {label_list[index]} Prediction: {pred_list[index]}')
        plt.axis('off')
        image = np.array(missclassified_image_list[index].cpu())
        image = np.transpose(image, (1, 2, 0))
        image = inv_transform(image=image)['image']
        plt.imshow(image)