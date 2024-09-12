from torch.utils.data import Dataset, DataLoader
import cv2
from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms

class SegmentationDataSet(Dataset):
    CLASSES = {"Write u classes RGB color like : 'blackground' : (0,0,0) " }

    def __init__(self, image_dir, mask_dir, classes=None, class_id=None, augmentation=None, preprocessing=None):
        self.ids = os.listdir(image_dir)
        self.images_fps = []
        for image_id in self.ids:
            image_path = os.path.join(image_dir, image_id)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image.shape[2] != 3:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            self.images_fps.append(image)
        self.mask_dir = mask_dir
        self.class_id = class_id
        self.class_values = [self.CLASSES[cls.lower()] for cls in classes] if classes else None
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        image_path = self.images_fps[i]
        try:
            image = Image.open(image_path)
            image = image.convert('RGB')
        except Exception as e:
            print(f"Error opening image {image_path}: {e}")
            return None

        if image.mode != 'RGB':
            raise ValueError('Input image should have 3 channels')

        if len(image.size) != 3 or image.size[2] != 3:
            raise ValueError('Input image should have 3 channels')

        if self.class_id is not None:
            mask_path = os.path.join(self.mask_dir, self.ids[i].replace('.jpg', f'_{self.class_id}.png'))
        else:
            mask_path = os.path.join(self.mask_dir, self.ids[i].replace('.jpg', '_mask.png'))

        mask = Image.open(mask_path)
        mask = mask.convert('RGB')
        mask = mask.resize((512, 512))
        mask = np.array(mask)
        mask = mask.astype(float) / 255.0
        image_size = 512
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])
        image = transform(image)

        if self.augmentation:
            sample = {'image': image.numpy(), 'mask': mask.numpy()}
            sample = self.augmentation(**sample)
            image, mask = sample['image'], sample['mask']

        if self.class_values:
            mask = mask.astype('float')

        if self.preprocessing:
            sample = {'image': image, 'mask': mask}
            sample = self.preprocessing(**sample)
            image, mask = sample['image'], sample['mask']

        if mask.ndim == 2:
            mask = np.tile(mask[..., np.newaxis], (1, 1, 3))
        elif mask.shape[-1] != 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

        return image, mask

    def __len__(self):
        return len(self.ids)
