import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time

from visualizing import get_model_name


def determine_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("CUDA is available. Using GPU for training.")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU for training.")

    return device

def evaluate(net, data_loader, criterion, device):
    total_loss = 0.0
    total_err = 0.0
    total_samples = 0

    # move the model to the correct device
    net.to(device)

    net.eval() # Set model to evaluation mode for optimization
    with torch.no_grad(): # Disable gradient calculation for optimization
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = net(inputs)
            loss = criterion(outputs, labels)

            # Multi-class error computation
            _, predicted = torch.max(outputs, 1)  # Get index of highest logit
            total_err += (predicted != labels).sum().item()
            total_loss += loss.item()
            total_samples += labels.size(0)

    err = float(total_err) / total_samples
    loss = float(total_loss) / (i + 1)
    return err, loss

def train_net(net, train_loader, device, val_loader=None, batch_size=64,
              learning_rate=0.01, num_epochs=30):
    torch.manual_seed(1000)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)

    # only init validation error/loss variables if a val_loader is provided
    if (val_loader is not None):
        val_err = np.zeros(num_epochs)
        val_loss = np.zeros(num_epochs)

    # move the model to the correct device
    net.to(device)

    print("Beginning Training...")
    start_time = time.time()
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        total_train_loss = 0.0
        total_train_err = 0.0
        total_samples = 0

        net.train() # Set model to training mode
        for i, data in enumerate(train_loader, 0):
            # Get the inputs, and move them to the right hardware (device)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass, backward pass, and optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Calculate error and store it for multi class classification
            _, predicted = torch.max(outputs, 1)
            total_train_err += (predicted != labels).sum().item()

            # Store loss
            total_train_loss += loss.item()
            total_samples += len(labels)

        train_err[epoch] = float(total_train_err) / total_samples
        train_loss[epoch] = float(total_train_loss) / (i+1)

        # Only calculate valdation error/loss if a val_loader is provided
        if (val_loader is not None):
            val_err[epoch], val_loss[epoch] = evaluate(net, val_loader,
                                                       criterion, device)
            print(
                f"Epoch: {epoch + 1} "
                f"Train err: {train_err[epoch]}, "
                f"Train loss: {train_loss[epoch]} "
                f"| Validation err: {val_err[epoch]}, "
                f"Validation loss: {val_loss[epoch]}"
            )
        else:
            print(
                f"Epoch: {epoch + 1} "
                f"Train err: {train_err[epoch]}, "
                f"Train loss: {train_loss[epoch]}"
            )

        # Save the current model (checkpoint) to a file
        model_path = get_model_name(net.name, batch_size, learning_rate, epoch)
        torch.save(net.state_dict(), model_path)

    print('Ending Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    # Write the train loss/err into CSV file for plotting later
    np.savetxt("{}_train_err.csv".format(model_path), train_err)
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)

    # Only save the validation error/loss if a val_loader is provided
    if (val_loader is not None):
        np.savetxt("{}_val_err.csv".format(model_path), val_err)
        np.savetxt("{}_val_loss.csv".format(model_path), val_loss)

    # at the very end, empty the cuda cache if cuda is used
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
