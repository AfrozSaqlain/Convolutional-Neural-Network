from pathlib import Path
from PIL import Image
import random
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm # type: ignore
from typing import Dict, Tuple, List
import seaborn as sns # type: ignore
from mlxtend.plotting import plot_confusion_matrix # type: ignore

import torch # type: ignore
import torchinfo # type: ignore
import torchvision # type: ignore 
from torchinfo import summary # type: ignore
import torchvision.models as models  # type: ignore
from torchvision.models import VGG19_Weights # type: ignore
from torch import nn # type: ignore
from torch.utils.data import DataLoader # type: ignore
from torchvision import datasets, transforms # type: ignore
from sklearn.metrics import confusion_matrix  # type: ignore
from sklearn import metrics # type: ignore
from torchmetrics import ConfusionMatrix # type: ignore


## Redirecting output to txt file
sys.stdout = open("log.out", "w")
sys.stderr = open("error.err", "w")

## Accessing directories
train_dir = Path("training")
test_dir = Path("testing")

## Transformation applied to images
data_transform = transforms.Compose([
    transforms.Resize(size=(300, 600)),
    transforms.ToTensor()
])

## Converting images to Datasets
train_data = datasets.ImageFolder(root = train_dir,
                                  transform = data_transform,
                                  target_transform = None)

test_data = datasets.ImageFolder(root = test_dir,
                                  transform = data_transform,
                                  target_transform = None)


print(f'Training dataset has following classes: {train_data.classes}')
print(f'Testing dataset has following classes: {test_data.classes}')
print('Training data dictionary', train_data.class_to_idx)

print(f'Length of train data {len(train_data)}, and test data: {len(test_data)}')

## Converting loaded images into Dataloaders
BATCH_SIZE = 50
train_dataloader = DataLoader(dataset = train_data,
                              batch_size = BATCH_SIZE,
                              num_workers = os.cpu_count(),
                              shuffle = True)    

test_dataloader = DataLoader(dataset = test_data,
                             batch_size = BATCH_SIZE,
                             num_workers = os.cpu_count(),
                             shuffle = False)

print(f'Length of train dataloader: {len(train_dataloader)} and test dataloader: {len(test_dataloader)}')

## Create TinyVGG model class
device = "cuda" if torch.cuda.is_available() else "cpu" 

all_preds, all_labels = [], []

# Create train step
def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device = device):
  model.train()

  # Setup trian loss and train accuracy values
  train_loss, train_acc = 0, 0

  # Loop through dta loader data batches
  for batch, (X, y) in enumerate(dataloader):
    # Send data to target device
    X, y = X.to(device), y.to(device)

    y_pred = model(X)
    y_pred_lables = torch.argmax(torch.softmax(y_pred, dim = 1), dim = 1)

    # print('\nTraining Preds:\n', y_pred_lables, '\n')

    loss = loss_fn(y_pred, y)
    train_loss += loss.item()

    optimizer.zero_grad()

    loss.backward()

    optimizer.step()

    y_pred_class = torch.argmax(torch.softmax(y_pred, dim = 1), dim = 1)
    train_acc += (y_pred_class == y).sum().item() / len(y_pred)

  # Adjust metrics to get average loss and accuracy per batch
  train_loss = train_loss / len(dataloader)
  train_acc = train_acc / len(dataloader)
  return train_loss, train_acc

# Create test step
def test_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               confusion_matrix_stat: bool,
               device = device):
  model.eval()

  # Setup trian loss and train accuracy values
  test_loss, test_acc = 0, 0

  # Turn on inference mode
  with torch.inference_mode():
    # Loop through dta loader data batches
    for batch, (X, y) in enumerate(dataloader):
      # Send data to target device
      X, y = X.to(device), y.to(device)

      test_pred_logits = model(X)

      loss = loss_fn(test_pred_logits, y)

      scheduler.step()

      test_loss += loss.item()

      test_pred_labels = torch.argmax(torch.softmax(test_pred_logits, dim = 1), dim = 1)
      
      if confusion_matrix_stat:
          all_preds.extend(test_pred_labels.cpu().numpy())
          all_labels.extend(y.cpu().numpy())

      test_acc += (test_pred_labels == y).sum().item() / len(test_pred_logits)

  # Adjust metrics to get average loss and accuracy per batch
  test_loss = test_loss / len(dataloader)
  test_acc = test_acc / len(dataloader)
  return test_loss, test_acc


def train_test(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5,
          device = device):

  # 1. Create an empty results dictionary
  results = {"train loss": [], "train acc": [], "test loss": [], "test acc": []}

  # 2. Loop through trainig and testing steps for a number of epochs
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model = model,
                                      dataloader = train_dataloader,
                                      loss_fn = loss_fn,
                                      optimizer = optimizer,
                                      device = device)
    if epoch == epochs - 1:
       confusion_matrix_stat = True
    else:
       confusion_matrix_stat = False
    test_loss, test_acc = test_step(model = model,
                                    dataloader = test_dataloader,
                                    loss_fn = loss_fn,
                                    confusion_matrix_stat = confusion_matrix_stat,
                                    device = device)

    # 3. Print out what's happening
    print(f"Epoch: {epoch} | Train loss: {train_loss:.4f}, Train accuracy: {train_acc:.4f} | Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

    # 4. Ipdate results dictionary
    results["train loss"].append(train_loss)
    results["test loss"].append(test_loss)
    results["train acc"].append(train_acc)
    results["test acc"].append(test_acc)

  # 4. Return the filled results at the end of the epochs
  return results

# Set number of epochs
NUM_EPOCHS = 100

## Defining my own learning rate scheduler
class CosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1):
        """
        Args:
            optimizer: Wrapped optimizer.
            T_max: Maximum number of iterations (or epochs).
            eta_min: Minimum learning rate.
            last_epoch: The index of the last epoch. Default: -1
        """
        self.T_max = T_max
        self.eta_min = eta_min
        super(CosineAnnealingLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            self.eta_min + (base_lr - self.eta_min) * (1 + np.cos(np.pi * self.last_epoch / self.T_max)) / 2
            for base_lr in self.base_lrs
        ]

# Load pre-trained VGG-19 model
vgg19 = models.vgg19(weights=VGG19_Weights.DEFAULT)

# Freeze convolutional layers
for param in vgg19.features.parameters():
    param.requires_grad = False

# Modify the classifier for two-class classification
num_features = vgg19.classifier[6].in_features
vgg19.classifier[6] = torch.nn.Linear(num_features, len(train_data.classes))  # Adjust for your number of classes

# Move model to device
vgg19.to(device)


# # Define the VGG-19 model architecture
# vgg19 = models.vgg19(weights=None)  # No pre-trained weights, since we'll load our own

# # Load your saved model state dict
# model_path = "vgg19_model.pth"  # Replace with your actual path
# vgg19.load_state_dict(torch.load(model_path))

# # # Set model to evaluation mode (optional, for inference)
# # vgg19.eval()

# # Freeze convolutional layers
# for param in vgg19.features.parameters():
#     param.requires_grad = False

# # Modify the classifier for two-class classification
# num_features = vgg19.classifier[6].in_features
# vgg19.classifier[6] = torch.nn.Linear(num_features, 2)  # Change 2 to match your number of classes

# # Move to GPU if available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# vgg19.to(device)



# Setup loss function and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params = vgg19.parameters(),
                             lr = 0.001)
scheduler = CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

# Start the timer
from timeit import default_timer as timer
start = timer()

# Train model_0
model_0_results = train_test(model = vgg19,
                        train_dataloader = train_dataloader,
                        test_dataloader = test_dataloader,
                        optimizer = optimizer,
                        loss_fn = loss_fn,
                        epochs = NUM_EPOCHS)

# End the timer
end = timer()

print(f"Total training time: {(end - start):.3f}")


def plot_loss_curves(results: Dict[str, List[float]]):
  # Get the loss values of results dictionary (training and testing set)
  loss = results["train loss"]
  test_loss = results["test loss"]

  acc = results["train acc"]
  test_acc = results["test acc"]

  epochs = range(len(results["train loss"]))

  plt.figure(figsize=(15,7))
  plt.subplot(1, 2, 1)
  plt.plot(epochs, loss, label = 'train loss')
  plt.plot(epochs, test_loss, label = 'test loss')
  plt.title("Loss Curve")
  plt.xlabel("Epochs")
  plt.ylabel("Loss")
  plt.legend()

  plt.subplot(1, 2, 2)
  plt.plot(epochs, acc, label = 'train accuracy')
  plt.plot(epochs, test_acc, label = 'test accuracy')
  plt.title("Accuracy Curve")
  plt.xlabel("Epochs")
  plt.ylabel("Accuracy")
  plt.legend()
  
  plt.savefig('Loss_and_Accuracy_curve.png')

plot_loss_curves(model_0_results)

torch.save(vgg19.state_dict(), 'new_vgg19_model.pth')



# Setup confusion instance and compare perdictions to targets
confmat = ConfusionMatrix(num_classes = len(train_data.classes), task = "multiclass")
confmat_tensor = confmat(preds = torch.tensor(all_preds),
                         target = torch.tensor(all_labels))

# Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat = confmat_tensor.numpy(),
    class_names = train_data.classes,
    figsize = (10, 8)
)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.savefig('New_Confusion_Matrix.png')
plt.close()


sys.stdout.close()
sys.stderr.close()