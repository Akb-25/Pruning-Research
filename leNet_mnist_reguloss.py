
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.utils.prune as prune
from torch.utils.tensorboard import SummaryWriter  # Import SummaryWriter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
from thop import profile, clever_format

#%load_ext tensorboard


batch_size = 64
num_classes = 10
learning_rate = 0.001
num_epochs = 10
# data transformtaion pipeline
# Normalizes the tensor by subtracting the mean (0.5) and dividing by the standard deviation 
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
# nn.Module, which is the base class for all neural network modules in PyTorch
# LeNet-5 is a classic convolutional neural network (CNN) architecture
#  Convolutional layers apply a set of learnable filters (kernels) to the input image, extracting features through convolution operations
# Filters are small, learnable matrices applied to input data (such as images) in convolutional layers.
# Each filter extracts specific features from the input data by performing convolution operations.
# Filters are small, learnable matrices applied to input data (such as images) in convolutional layers.
# Each filter extracts specific features from the input data by performing convolution operations.
# Channels refer to the depth dimension of the input data or feature maps in convolutional layers.
# In grayscale images, there is typically one channel representing the intensity values of pixels.
# In color images represented in the RGB format, there are three channels: one for red, one for green, and one for blue.
#CNNs, filters convolve with input data across multiple channels to produce feature maps capturing various visual patterns and characteristics.
# Linear Layers These layers are typically used to learn complex patterns by combining features learned from convolutional layers
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
# x typically represents the input image or a batch of input images
    def forward(self, x):
        # The output of the convolution operation is passed through the rectified linear unit (ReLU) activation function (F.relu()), which introduces non-linearity to the network.
        x = F.relu(self.conv1(x))
        # Max pooling reduces the spatial dimensions of the feature maps while retaining the most important information.
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # Reshapes (flattens) the output tensor x into a 1D tensor.
        x = x.view(x.size(0), -1)
        # Applies the first fully connected (linear) layer (self.fc1) to the flattened input tensor x. The output is passed through the ReLU activation function.
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # The output tensor containing the raw scores/logits for each class
        x = self.fc3(x)
        return x
# Regularization is a technique used in machine learning and deep learning to prevent overfitting. Regularization methods introduce additional constraints or penalties to the learning process to encourage simpler models that generalize better
# The regularization term is an additional component added to the loss function during training to penalize complex models
def regularization(model, lambda_):
    regu = 0.0
    for param in model.parameters():
        regu += torch.sum(param**2)  # L2-norm
       # regu -= torch.sum(param**2)  # Change the sign to maximize values
    return 0.5 * lambda_ * regu  # Multiply by 0.5 to match the derivative of L2-norm
# Generalization refers to the ability of a machine learning model to perform well on unseen data, beyond the training dataset
def calculate_pruning_ratio(model):
    total_params = 0
    pruned_params = 0
# Iterates through all modules (layers) of the neural network model.Named_modules() is a method provided by PyTorch that returns an iterator over all modules in the model, along with their names.
    for name, module in model.named_modules():
        # Checks if the current module is an instance of the nn.Conv2d class.
        if isinstance(module, nn.Conv2d):
            # Computes the total number of parameters in the current convolutional layer and adds it to the total_params variable.
            total_params += module.weight.nelement()
            # Counts the number of pruned parameters in the current convolutional layer and adds it to the pruned_params variable.
            #torch.sum(module.weight == 0) calculates the number of elements in the weight tensor that are equal to zero, indicating pruned connections.
            pruned_params += torch.sum(module.weight == 0).item()

    pruning_ratio = pruned_params / total_params
    return pruning_ratio
# This function, calculate_flops, is designed to estimate the number of floating-point operations (FLOPs) required to execute the given neural network model on a specific input tensor. 
def calculate_flops(model, input_tensor):
    # model.eval(): Puts the model in evaluation mode. This is typically done to disable dropout and batch normalization layers during inference, ensuring consistent behavior between training and inference.
    model.eval()
    # Uses the profile function to profile the model's computational complexity in terms of FLOPs and the number of parameters.
    flops, params = profile(model, inputs=(input_tensor,), verbose=False)
    # Formats the estimated number of FLOPs (flops) into a human-readable string.
    flops_str = clever_format([flops], "%.2f")[0]
    return flops_str

# Temporarily disables gradient calculation to reduce memory consumption and speed up computation during evaluation
def evaluate(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            # Moves the input images and labels to the specified device (CPU or GPU) for computation.
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            # : Extracts the predicted class labels by taking the index of the maximum value along the predicted output probabilities.
            _, predicted = torch.max(outputs.data, 1)
            #  Increments the total count by the batch size 
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy


# Example usage:

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Generates a random tensor of shape (1, 1, 28, 28).
# The first dimension (1) represents the batch size (a single image in this case).
# The second dimension (1) represents the number of channels (grayscale image, so one channel).
# The last two dimensions (28x28) represent the height and width of the image, respectively.
input_tensor = torch.randn(1, 1, 28, 28).to(device)  # Adjust the input shape


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = LeNet5().to(device)
# This loss function combines softmax activation and cross-entropy loss, making it suitable for multi-class classification problems.
# CE=−∑iY^i log(pi)
# The loss function penalizes incorrect predictions more heavily, especially when the predicted probability diverges significantly from the true probability.
cost = nn.CrossEntropyLoss()
# Initializes the optimizer for updating the model parameters during training.
# Optimizers are algorithms used to update the parameters (weights and biases) of a neural network during training in order to minimize the loss function.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs_list = []
train_loss_list = []
train_accuracy_list = []  # Added to store training accuracy
valid_loss_list = []
valid_accuracy_list = []

valid_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform), batch_size=batch_size, shuffle=False)



# Calculate pruning ratio and FLOPs before pruning
pruning_ratio_before = calculate_pruning_ratio(model)
flops_before_pruning = calculate_flops(model, input_tensor)

print(f"Pruning Ratio before pruning: {pruning_ratio_before:.4f}")
print(f"FLOPs before pruning: {flops_before_pruning}")

# Learning rate scheduler
# A learning rate scheduler is a tool used in training neural networks to adjust the learning rate during optimization dynamically.
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Create a SummaryWriter
writer = SummaryWriter()

best_model = None
best_accuracy = 0.0
lambda_ = 0.001  # Regularization strength

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    running_reg_loss = 0.0  # New variable to accumulate regularization loss
    total_reg_value = 0.0
    correct_train = 0  # Initialize correct_train
    total_train = 0  # Initialize total_train

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        # Clears the gradients of all optimized parameters before performing a backward pass. 
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = cost(outputs, labels)
        reg_loss = regularization(model, lambda_)
        regularization_=2 * (1/lambda_) *reg_loss
        total_loss = loss + reg_loss

        total_loss.backward()
        # Updates the model parameters using the computed gradients and the optimization algorithm specified by the optimizer
        optimizer.step()

        running_loss += loss.item()
        running_reg_loss += reg_loss.item()

         # Calculate training accuracy
        _, predicted_train = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted_train == labels).sum().item()
         # Log training loss, regularization loss, and total regularization value
        writer.add_scalar('Loss/train', running_loss / len(train_loader), epoch * len(train_loader) + i)
        writer.add_scalar('Regularization_Loss/train', running_reg_loss / len(train_loader), epoch * len(train_loader) + i)
        writer.add_scalar('Total_Regularization_Value/train', total_reg_value, epoch * len(train_loader) + i)

    valid_accuracy = evaluate(model, valid_loader, device)

    # Calculate and store training accuracy
    train_accuracy = 100 * correct_train / total_train
    train_accuracy_list.append(train_accuracy)
    # Update the learning rate scheduler
    scheduler.step()

    # Calculate regularization value (not accumulated)
    total_reg_value = sum(torch.sum(torch.abs(param)) for param in model.parameters())
    valid_accuracy = evaluate(model, valid_loader, device)
    epochs_list.append(epoch + 1)
    train_loss_list.append(running_loss / len(train_loader))
    valid_loss_list.append(loss.item())
    valid_accuracy_list.append(valid_accuracy)

     # Log validation accuracy
    writer.add_scalar('Accuracy/validation', valid_accuracy, epoch)
     # Log training accuracy
    writer.add_scalar('Accuracy/train', train_accuracy, epoch)
    writer.add_scalar('Learning_Rate', scheduler.get_lr()[0], epoch)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Loss: {running_loss/len(train_loader):.4f}, "
          f"Regularization Loss: {running_reg_loss/len(train_loader):.4f}, "
          f"regu:{regularization_},"
          f"Total Regularization Value: {total_reg_value:.4f}, "
          f"Training Accuracy: {100 * correct_train / total_train:.2f}%, "
          f"Validation Accuracy: {valid_accuracy:.2f}%")

    # Save the model if the current accuracy is higher than the best accuracy
    if valid_accuracy > best_accuracy:
        best_accuracy = valid_accuracy
        # This line saves the state of the model's parameters (weights and biases) as a dictionary.
        best_model = model.state_dict()

data = {
    'Epoch': epochs_list,
    'Training Loss': train_loss_list,
    'Validation Accuracy': valid_accuracy_list
}

df = pd.DataFrame(data)
df.to_excel('training_metrics.xlsx', index=False)

# Print training and validation accuracy
print(f"Final Training Accuracy: {train_accuracy_list[-1]:.2f}%")
print(f"Final Validation Accuracy: {valid_accuracy_list[-1]:.2f}%")

# After training, visualize the model graph in TensorBoard
model_input = torch.randn(1, 1, 28, 28).to(device)  # Adjust the input shape
# This line adds the computational graph of the model to the TensorBoard log file created by the SummaryWriter
writer.add_graph(model, model_input)

# Close the SummaryWriter
writer.close()

# Load the best model
# It restores the model to the state it was in when best_model was saved, allowing you to continue training or use the best model for evaluation or inference.
model.load_state_dict(best_model)

# Evaluate the best model on the test set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data[0].to(device), data[1].to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Best Model Accuracy on Test Set: {100 * correct / total:.2f}%")
print("Best Accuracy of model: ", best_accuracy)


# Prune the model (you can adapt this based on your pruning method)
# This line applies L1 unstructured pruning to the weights of the conv1 layer of the model
prune.l1_unstructured(model.conv1, name="weight", amount=0.2)
# Calculate pruning ratio and FLOPs after pruning
pruning_ratio_after = calculate_pruning_ratio(model)
flops_after_pruning = calculate_flops(model, input_tensor)

print(f"Pruning Ratio after pruning: {pruning_ratio_after:.4f}")
print(f"FLOPs after pruning: {flops_after_pruning}")


# Function to prune the top k filters with least L1-norm

def prune_top_k_filters(module, name, amount):
    if isinstance(module, nn.Conv2d) and name == 'weight':
        # Checks if the provided module is an instance of the nn.Conv2d class (convolutional layer) and if the parameter to be pruned is the weight
        weight = module.weight.data
        # computes the L1-norm of the weight tensor along dimensions (1, 2, 3), which correspond to the height, width, and channels of the filters.
        # This calculates the sum of absolute values for each filter's weights, resulting in a 1D tensor containing the L1-norm for each filter.
        l1_norms = torch.norm(weight, p=1, dim=(1, 2, 3))  # Calculate L1-norm for each filter
        #  Finds the indices of the top-k smallest L1-norms.
        # k is determined by multiplying the specified amount with the total number of filters 
        _, indices = torch.topk(l1_norms, k=int(amount * weight.shape[0]), largest=False)
        # k is determined by multiplying the specified amount with the total number of filters 
        # This function effectively removes the selected filters by zeroing out their weights, thus pruning them from the model.
        # mask parameter is used to specify which parameters (weights) of the module should be pruned
        # The mask argument should be a tensor of the same shape as the parameter being pruned, where a value of 1 indicates that the corresponding parameter should be kept, and a value of 0 indicates that the parameter should be pruned.
        prune.custom_from_mask(module, name='weight', mask=indices)

# Prune the model by specifying the top k filters to keep
k = 10  # Adjust the value of k as needed
for name, module in model.named_modules():
    # adjusts the pruning amount based on the number of output features of the fully connected layer fc1
    # k, you determine how many of the top filters with the smallest L1-norms are retained
    prune_top_k_filters(module, name, amount=k / model.fc1.out_features)  # Adjust the amount for each layer


# Prune the model by specifying the top k filters to keep
k = 10  # Adjust the value of k as needed
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        # It applies L1 unstructured pruning to each convolutional layer, removing 20% of the weights with the smallest absolute values.
        prune.l1_unstructured(module, name="weight", amount=0.2)  # Prune 20% of weights

# Remove pruning from the entire model before saving
for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d):
        prune.remove(module, name="weight")

# Save the pruned model
torch.save(model.state_dict(), 'pruned_lenet5_mnist.pth')

# After training, load the best model for retraining or further fine-tuning
model.load_state_dict(best_model)

# Continue with retraining or fine-tuning using the best_model

# After training, load the best model for retraining or further fine-tuning
model.load_state_dict(best_model)

# Define a new optimizer and criterion for retraining/fine-tuning
optimizer_retrain = torch.optim.Adam(model.parameters(), lr=learning_rate)  # Adjust the learning rate if needed
criterion_retrain = nn.CrossEntropyLoss()  # Define a new criterion if needed

num_retrain_epochs = 5  # Adjust the number of retraining epochs as needed

# Retraining loop
for epoch in range(num_retrain_epochs):
    model.train()
    running_loss = 0.0
    running_reg_loss = 0.0  # New variable to accumulate regularization loss

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer_retrain.zero_grad()
        outputs = model(inputs)
        loss = criterion_retrain(outputs, labels)
        reg_loss = regularization(model, lambda_)
        regularization_=2 * (1/lambda_) *reg_loss
        total_loss = loss + reg_loss

        total_loss.backward()
        optimizer_retrain.step()

        running_loss += loss.item()
        running_reg_loss += reg_loss.item()

    # Evaluate the retrained model on the validation set
    valid_accuracy_retrain = evaluate(model, valid_loader, device)

    # Calculate regularization value (not accumulated)
    total_reg_value_retrain = sum(torch.sum(torch.abs(param)) for param in model.parameters())

    epochs_list.append(epoch + 1)
    train_loss_list.append(running_loss / len(train_loader))
    valid_accuracy_list.append(valid_accuracy_retrain)

    print(f"Retrain Epoch [{epoch+1}/{num_retrain_epochs}], "
          f"Loss: {running_loss/len(train_loader):.4f}, "
          f"Regularization Loss: {running_reg_loss/len(train_loader):.4f}, "
          f"regu:{regularization_},"
          f"Total Regularization Value: {total_reg_value_retrain:.4f}, "
          f"Validation Accuracy: {valid_accuracy_retrain:.2f}%")

# Calculate pruning ratio and FLOPs after retraining
pruning_ratio_after_retrain = calculate_pruning_ratio(model)
flops_after_retrain = calculate_flops(model, input_tensor)

print(f"Pruning Ratio after retraining: {pruning_ratio_after_retrain:.4f}")
print(f"FLOPs after retraining: {flops_after_retrain}")


    # After retraining, you can further fine-tune as needed
# Define a new optimizer and criterion for fine-tuning if needed
optimizer_finetune = torch.optim.Adam(model.parameters(), lr=learning_rate / 10)  # Adjust the learning rate if needed
criterion_finetune = nn.CrossEntropyLoss()  # Define a new criterion if needed

num_finetune_epochs = 5  # Adjust the number of fine-tuning epochs as needed

# Fine-tuning loop
for epoch in range(num_finetune_epochs):
    model.train()
    running_loss = 0.0
    running_reg_loss = 0.0  # New variable to accumulate regularization loss

    for i, data in enumerate(train_loader, 0):
        inputs, labels = data[0].to(device), data[1].to(device)
        optimizer_finetune.zero_grad()
        outputs = model(inputs)
        loss = criterion_finetune(outputs, labels)
        reg_loss = regularization(model, lambda_)
        regularization_=2 * (1/lambda_) *reg_loss
        total_loss = loss + reg_loss

        total_loss.backward()
        optimizer_finetune.step()

        running_loss += loss.item()
        running_reg_loss += reg_loss.item()

    # Evaluate the fine-tuned model on the validation set
    valid_accuracy_finetune = evaluate(model, valid_loader, device)

    # Calculate regularization value (not accumulated)
    total_reg_value_finetune = sum(torch.sum(torch.abs(param)) for param in model.parameters())

    epochs_list.append(epoch + 1)
    train_loss_list.append(running_loss / len(train_loader))
    valid_accuracy_list.append(valid_accuracy_finetune)

    print(f"Fine-tune Epoch [{epoch+1}/{num_finetune_epochs}], "
          f"Loss: {running_loss/len(train_loader):.4f}, "
          f"Regularization Loss: {running_reg_loss/len(train_loader):.4f}, "
          f"regu:{regularization_},"
          f"Total Regularization Value: {total_reg_value_finetune:.4f}, "
          f"Validation Accuracy: {valid_accuracy_finetune:.2f}%")

# Calculate pruning ratio and FLOPs after fine-tuning
pruning_ratio_after_finetune = calculate_pruning_ratio(model)
flops_after_finetune = calculate_flops(model, input_tensor)

print(f"Pruning Ratio after fine-tuning: {pruning_ratio_after_finetune:.4f}")
print(f"FLOPs after fine-tuning: {flops_after_finetune}")

# Plotting the graph
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(epochs_list, train_loss_list, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_list, valid_accuracy_list, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

#%tensorboard --logdir=runs