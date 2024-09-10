import os
import cv2


def augmentation():
    input_dir = "train"
    output_dir_tiff = "division_tiff"
    output_dir_mask = "division_mask"

    if not os.path.exists(output_dir_tiff):
        os.mkdir(output_dir_tiff)
    if not os.path.exists(output_dir_mask):
        os.mkdir(output_dir_mask)

    for file in os.listdir(input_dir):
        if file.endswith(".jpg") or file.endswith(".png") or file.endswith(".tiff"):
            img = cv2.imread(os.path.join(input_dir, file))

            output_dir = os.path.join(output_dir_mask if file.endswith(".png") else output_dir_tiff,
                                      os.path.splitext(file)[0])
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            img_rotate = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

            img_rotate_256 = cv2.resize(img_rotate, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            for x in range(0, img_rotate_256.shape[1], 512):
                for y in range(0, img_rotate_256.shape[0], 512):
                    img_block = img_rotate_256[y:y + 512, x:x + 512]
                    if img_block.shape[0] == img_block.shape[1] == 512:  # Check if the image is 512x512
                        cv2.imwrite(
                            os.path.join(output_dir, f'{os.path.splitext(file)[0]}_rotate_{x // 512}_{y // 512}.jpg'),
                            img_block)

            img_zoom = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

            img_zoom_256 = cv2.resize(img_zoom, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            for x in range(0, img_zoom_256.shape[1], 512):
                for y in range(0, img_zoom_256.shape[0], 512):
                    img_block = img_zoom_256[y:y + 512, x:x + 512]
                    if img_block.shape[0] == img_block.shape[1] == 512:
                        cv2.imwrite(
                            os.path.join(output_dir, f'{os.path.splitext(file)[0]}_zoom_{x // 512}_{y // 512}.jpg'),
                            img_block)

            img_flip = cv2.flip(img, 0)

            img_flip_512 = cv2.resize(img_flip, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            for x in range(0, img_flip_512.shape[1], 512):
                for y in range(0, img_flip_512.shape[0], 512):
                    img_block = img_flip_512[y:y + 512, x:x + 512]
                    if img_block.shape[0] == img_block.shape[1] == 512:
                        cv2.imwrite(
                            os.path.join(output_dir, f'{os.path.splitext(file)[0]}_flip_{x // 512}_{y // 512}.jpg'),
                            img_block)
