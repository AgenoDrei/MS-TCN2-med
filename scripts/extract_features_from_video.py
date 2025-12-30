from fractions import Fraction
import os
import click
import av
import time
import torch
import cv2
import math
import timm
import numpy as np
from transformers import pipeline
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

TMP_DIR = "/tmp/video_processing"


class MyDataset(Dataset):
    def __init__(self, path, target_fps, width=480, height=270):
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]

        self.input_fps = float(self.stream.average_rate)
        self.skip_factor = math.ceil(self.input_fps / target_fps)
        self.width = width
        self.height = height
        print(f"Input FPS: {self.input_fps}, Target FPS: {target_fps}, Skip Factor: {self.skip_factor}, Size: {self.width}x{self.height}")

    def __len__(self):
        return int(self.container.streams.video[0].frames // self.skip_factor)

    def get_data(self):
        in_frame_count = 0
        yielded_count = 0
        for frame in self.container.decode(video=0):
            # Frame rate reduction logic
            if in_frame_count % self.skip_factor == 0:
                if self.height != frame.height or self.width != frame.width:
                    reduced_frame = frame.reformat(width=self.width, height=self.height, format='rgb24')
                else:
                    reduced_frame = frame
                yielded_count += 1
                yield reduced_frame.to_image()
            
            in_frame_count += 1
    
    def __del__(self):
        self.container.close()


@click.command()
@click.argument("input_path", type=click.Path(exists=True, dir_okay=False))
@click.option("--output", "-o", type=click.Path(), default=None, help="Output .npy file path")
@click.option("--device", type=int, default=0, help="Device for pipeline (0 for first GPU, -1 for CPU)")
def cli(input_path, output, device): 
    # Step 2: Extract features using Huggingface pipeline
    print(f"Video resized and resampled, time: {time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}")
    dataset = MyDataset(input_path, target_fps=5)
    batch_size = 256

    pipe = pipeline("image-feature-extraction", model="timm/vit_large_patch16_dinov3.lvd1689m", device=device)
    features = []
    for feat in tqdm(pipe(dataset.get_data(), batch_size=batch_size, return_tensors="np"), total=len(dataset)):
        features.append(feat.squeeze().mean(axis=1))

    # Step 3: Save features to .npy file
    print(f"Features extracted, time: {time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}")
    vec = np.array(features)
    if output:
        np.save(output, vec)
    click.echo(f"Saved features to {output} (shape: {vec.shape})")

if __name__ == "__main__":
    os.makedirs(TMP_DIR, exist_ok=True)
    cli()
