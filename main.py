import cv2
import torch
import torch.optim as optim
import os
from pathlib import Path
from torch import nn
from torch.utils.data import DataLoader, RandomSampler
from augmentation import augmentation
from filters import sobel , gaussian , line_detector
from filters import mean , edge_detector
from creating_dataset import SegmentationDataSet
from image_classification import ImageClassificationDataset
from Unet import UNet, UNetUpBlock , UNetConvBlock
from plot import
from eva_val_test_train import train , test , evaluate , validate


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device="cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR = './'

x_train_dir = os.path.join(DATA_DIR, 'train')
y_train_dir = os.path.join(DATA_DIR, 'trainannot')
x_valid_dir = os.path.join(DATA_DIR, 'val')
y_valid_dir = os.path.join(DATA_DIR, 'valannot')
x_test_dir = os.path.join(DATA_DIR, 'test')
y_test_dir = os.path.join(DATA_DIR, 'testannot')
x_division_dir = os.path.join(DATA_DIR, 'division_tiff')
y_division_dir = os.path.join(DATA_DIR, 'division_mask')

# Define directory for saving trained model
prepared_model_dir = os.path.join(DATA_DIR, 'prepared_model')
os.makedirs(prepared_model_dir, exist_ok=True)


def preprocessing(image):
    image = sobel(image)
    image = line_detector(image)
    image = edge_detector(image)
    image = mean(image)
    image = gaussian(image)

    image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

    return image

model = UNet().float().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss().to(device)

# Define data generators
train_dataset = SegmentationDataSet(
    x_division_dir, y_division_dir,
    classes=['field', 'forest', 'water', 'road', 'building', 'background'],
    augmentation=None,
    preprocessing=preprocessing
)

valid_dataset = SegmentationDataSet(
    x_valid_dir, y_valid_dir,
    classes=['field', 'forest', 'water', 'road', 'building', 'background'],
    augmentation=augmentation,
    preprocessing=preprocessing
)


test_dataset = ImageClassificationDataset(
    [os.path.join(x_test_dir, img) for img in os.listdir(x_test_dir)],
    classes=['field', 'forest', 'water', 'road', 'building', 'background']
)

def pixel_accuracy(outputs, targets):
    with torch.no_grad():
        predicted = torch.argmax(outputs, dim=1)
        correct = (predicted == targets).sum().item()
        total = targets.numel()
        accuracy = correct / total
        return accuracy


train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, drop_last=True)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=8)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)

# Define the training loop

# Train the model for a few epochs

#for epoch in range(3):
 #   train_loss = train(model, device, train_loader, optimizer, criterion, epoch+1)
  #  valid_loss = validate(model, device, valid_loader, criterion)
#test_loss = test(model, device, test_loader, criterion)


augmentation()
train(model, device, train_loader, optimizer,criterion,epoch=1 )