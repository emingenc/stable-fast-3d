import os
import torch
import time
import logging
from cog import BasePredictor, Input, Path

import rembg
from PIL import Image

from sf3d.system import SF3D
from sf3d.utils import remove_background, resize_foreground
from huggingface_hub import login

# Authenticate with Hugging Face
login(token='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')


def handle_image(image_path, idx, output_dir, rembg_session, foreground_ratio):
    image = remove_background(Image.open(image_path).convert("RGBA"), rembg_session)
    image = resize_foreground(image, foreground_ratio)
    os.makedirs(os.path.join(output_dir, str(idx)), exist_ok=True)
    path = os.path.join(output_dir, str(idx), "input.png")
    image.save(path)
    return path


class Predictor(BasePredictor):
    def setup(self) -> None:
        model = SF3D.from_pretrained(
            "stabilityai/stable-fast-3d",
            config_name="config.yaml",
            weight_name="model.safetensors",
        )
        device = torch.device("cpu")
        model.to(device)
        model.eval()
        self.model = model

    def predict(
        self,
        preview_img: Path = Input(description="The input image"),
        foreground_ratio: float = Input(
            default=0.5, description="The ratio of the foreground to the background"
        ),
        output_dir: Path = Input(description="The output directory"),
    ) -> None:
        start = time.time()
        rembg_session = rembg.new_session()

        image = handle_image(
            preview_img, 0, output_dir, rembg_session, foreground_ratio
        )
        texture_resolution = 1024
        with torch.no_grad():
            with torch.autocast():
                mesh, glob_dict = self.model.run_image(
                    image,
                    bake_resolution=texture_resolution,
                    remesh="none",
                )
        out_mesh_path = os.path.join(output_dir, "0", "mesh.glb")
        mesh.export(out_mesh_path, include_normals=True)

        stop = time.time()
        logging.info(f"Time taken: {stop - start} seconds")

        return out_mesh_path
