# import os
# import numpy as np
# import imageio
# from PIL import Image

# def sliding_window(image, step_size, window_size):
#     """Slide a window across the image."""
#     for y in range(0, image.shape[0], step_size):
#         for x in range(0, image.shape[1], step_size):
#             yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

# def pad_image(image, window_size):
#     """Pad the image with zeros to fit the window size."""
#     if len(image.shape) == 3:  # For RGB images
#         padded_image = np.zeros((max(image.shape[0], window_size[1]), max(image.shape[1], window_size[0]), image.shape[2]), dtype=image.dtype)
#     else:  # For grayscale images
#         padded_image = np.zeros((max(image.shape[0], window_size[1]), max(image.shape[1], window_size[0])), dtype=image.dtype)
#     padded_image[:image.shape[0], :image.shape[1]] = image
#     return padded_image

# def save_tiles(image, output_folder, window_size, step_size, prefix):
#     """Save the image tiles to the specified folder."""
#     padded_image = pad_image(image, window_size)
#     for (x, y, window) in sliding_window(padded_image, step_size, window_size):
#         if len(window.shape) == 3:
#             window = window[:window_size[1], :window_size[0], :]
#         else:
#             window = window[:window_size[1], :window_size[0]]
#         if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
#             if len(window.shape) == 3:
#                 window = np.pad(window, ((0, window_size[1] - window.shape[0]), (0, window_size[0] - window.shape[1]), (0, 0)), 'constant')
#             else:
#                 window = np.pad(window, ((0, window_size[1] - window.shape[0]), (0, window_size[0] - window.shape[1])), 'constant')
#         tile = Image.fromarray(window.astype(np.uint8))
#         tile.save(os.path.join(output_folder, f"{x//256}_{y//256}.png"))

# def process_image(image_path, output_folder, window_size=(256, 256), step_size=256, prefix='tile'):
#     image = imageio.imread(image_path)

#     if len(image.shape) == 2 and 'mask' not in image_path:  # Convert grayscale to RGB if not mask
#         image = np.stack((image,)*3, axis=-1)
#     if len(image.shape) == 3 and 'mask' in image_path:  # Convert RGB to grayscale if mask
#         image = image[:, :, 0]

#     os.makedirs(output_folder, exist_ok=True)

#     save_tiles(image, output_folder, window_size, step_size, prefix)

# # Example usage
# # process_image('/home/jicredt_data/dsj/CDdata/ROADCD/CRCD_src/after.tif', '/home/jicredt_data/dsj/CDdata/ROADCD/CRCD/A', prefix='A')
# # process_image('/home/jicredt_data/dsj/CDdata/ROADCD/CRCD_src/before.tif', '/home/jicredt_data/dsj/CDdata/ROADCD/CRCD/B', prefix='B')
# process_image('/home/jicredt_data/dsj/CDdata/ROADCD/CRCD_src/mask.png', '/home/jicredt_data/dsj/CDdata/ROADCD/CRCD/label', prefix='label')


import os
import numpy as np
import imageio
import cv2



def sliding_window(image, step_size, window_size):
    """Slide a window across the image."""
    for y in range(0, image.shape[0], step_size):
        for x in range(0, image.shape[1], step_size):
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

def pad_image(image, window_size):
    """Pad the image with zeros to fit the window size."""
    if len(image.shape) == 3:  # For RGB images
        padded_image = np.zeros((max(image.shape[0], window_size[1]), max(image.shape[1], window_size[0]), image.shape[2]), dtype=image.dtype)
    else:  # For grayscale images
        padded_image = np.zeros((max(image.shape[0], window_size[1]), max(image.shape[1], window_size[0])), dtype=image.dtype)
    padded_image[:image.shape[0], :image.shape[1]] = image
    return padded_image

def save_tiles(image, output_folder, window_size, step_size, prefix):
    """Save the image tiles to the specified folder."""
    padded_image = pad_image(image, window_size)
    for (x, y, window) in sliding_window(padded_image, step_size, window_size):
        if len(window.shape) == 3:
            window = window[:window_size[1], :window_size[0], :]
        else:
            window = window[:window_size[1], :window_size[0]]
        if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
            if len(window.shape) == 3:
                window = np.pad(window, ((0, window_size[1] - window.shape[0]), (0, window_size[0] - window.shape[1]), (0, 0)), 'constant')
            else:
                window = np.pad(window, ((0, window_size[1] - window.shape[0]), (0, window_size[0] - window.shape[1])), 'constant')
        # tile = Image.fromarray(window.astype(np.uint8))
        # tile.save(os.path.join(output_folder, f"{x//256}_{y//256}.png"))
        cv2.imwrite(os.path.join(output_folder, f"{x//256}_{y//256}.png"), window.astype(np.uint8))
        # tile.save(os.path.join(output_folder, f"{x//256}_{y//256}.png"))

def process_image(image_path, output_folder, window_size=(256, 256), step_size=256, prefix='tile'):
    # image = imageio.imread(image_path)
    image = cv2.imread(image_path, 0)

    if len(image.shape) == 2 and 'mask' not in image_path:  # Convert grayscale to RGB if not mask
        image = np.stack((image,)*3, axis=-1)
    if len(image.shape) == 3 and 'mask' in image_path:  # Convert RGB to grayscale if mask
        image = image[:, :, 0]

    os.makedirs(output_folder, exist_ok=True)

    save_tiles(image, output_folder, window_size, step_size, prefix)

# Example usage
# process_image('/home/jicredt_data/dsj/CDdata/ROADCD/CRCD_src/after.tif', '/home/jicredt_data/dsj/CDdata/ROADCD/CRCD/A', prefix='A')
# process_image('/home/jicredt_data/dsj/CDdata/ROADCD/CRCD_src/before.tif', '/home/jicredt_data/dsj/CDdata/ROADCD/CRCD/B', prefix='B')
process_image('/home/jicredt_data/dsj/CDdata/ROADCD/CRCD_src/mask.png', '/home/jicredt_data/dsj/CDdata/ROADCD/CRCD/label', prefix='label')
