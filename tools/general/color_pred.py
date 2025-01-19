import os
import cv2
import argparse
import tqdm


def process_images(input_dir):
    os.makedirs(input_dir+'_vis', exist_ok=True)
    files = os.listdir(input_dir)
    for file in tqdm.tqdm(files):
        img = cv2.imread(os.path.join(input_dir, file), 0)
        img_vis = img * 255
        cv2.imwrite(os.path.join(input_dir+'_vis', file), img_vis)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images using argparse")
    parser.add_argument("input_dir", help="Input directory containing images")
    args = parser.parse_args()

    process_images(args.input_dir)
