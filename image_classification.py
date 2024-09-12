from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch

class ImageClassificationDataset(Dataset):
    CLASSES = {"Write u classes like :' blackgound' : 0"}

    def __init__(self, image_paths, classes=None):
        self.image_paths = image_paths
        self.classes = [self.CLASSES[cls.lower()] for cls in classes] if classes else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)

        # Добавляем конвертацию изображения в RGB
        img = img.convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomResizedCrop(512),
            transforms.ToTensor(),
        ])
        img_tensor = transform(img)
        img_tensor = torch.transpose(img_tensor, 1, 2)
        img_tensor = torch.transpose(img_tensor, 0, 1)

        label = self.classes[idx] if self.classes else None
        return img_tensor, label
