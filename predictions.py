import cv2
import torch
import numpy as np
import os

def make_predictions(model, imagePath):
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient computation and memory tracking
    with torch.no_grad():
        # Load and preprocess the image
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype("float32") / 255.0
        image = cv2.resize(image, (128, 128))
        orig = image.copy()

        # Load the ground truth mask and resize it
        filename = imagePath.split(os.path.sep)[-1]
        MASK_DATASET_PATH = "trainannot"
        INPUT_IMAGE_HEIGHT = 256
        gtMask = cv2.imread(os.path.join(MASK_DATASET_PATH, filename), 0)
        gtMask = cv2.resize(gtMask, (INPUT_IMAGE_HEIGHT, INPUT_IMAGE_HEIGHT))

        # Transpose the image to match PyTorch format
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        image = torch.from_numpy(image).to(device)

        # Make predictions using the model
        predMask = model(image).squeeze()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()
        THRESHOLD = 0.5
        predMask = (predMask > THRESHOLD) * 255
        predMask = predMask.astype(np.uint8)

        # Display the results
        # Prepare the image for display
        gtMask = cv2.cvtColor(gtMask, cv2.COLOR_GRAY2RGB)
        gtMask = np.concatenate([gtMask, np.zeros_like(gtMask)], axis=-1)
        predMask = cv2.cvtColor(predMask, cv2.COLOR_GRAY2RGB)
        predMask = np.concatenate([np.zeros_like(predMask), predMask], axis=-1)
        display_image = np.concatenate([orig, gtMask, predMask], axis=1)

        # Display the image
        cv2.imshow("Prediction", display_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
