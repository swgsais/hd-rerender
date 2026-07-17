"""
Custom ComfyUI nodes for the SWG HD re-render pipeline's batched upscale.

Stock LoadImage loads exactly one image, and stock SaveImage only accepts a
shared filename_prefix plus its own auto-incrementing counter - there's no
way to save a batch back out under each slot's original filename. These two
nodes exist purely to fill that gap so hd_rerender.py can submit one
same-size batch of textures per /prompt call instead of one call per file.

Install: copy this file into ComfyUI/custom_nodes/, restart ComfyUI.

Written against the classic dict-based custom node API (INPUT_TYPES /
RETURN_TYPES / NODE_CLASS_MAPPINGS), not the newer io.ComfyNode schema API -
ComfyUI v0.27.0 still fully supports the classic style, and it's what
websocket_image_save.py (shipped in ComfyUI's own examples) uses for the
same batch-of-images-out pattern this borrows from.
"""
import os
import time

import numpy as np
import torch
from PIL import Image, ImageOps

import folder_paths


class SWGLoadImageBatch:
    """Load N named images from ComfyUI's input directory into one batch
    tensor. All images must share the same width/height - the caller is
    responsible for only grouping same-size textures into one call; this
    node fails loudly on a size mismatch rather than silently resizing,
    since that would mean the caller's bucketing logic has a bug.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "filenames": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "load_batch"
    CATEGORY = "swg_hd"

    def load_batch(self, filenames):
        names = [n.strip() for n in filenames.splitlines() if n.strip()]
        if not names:
            raise ValueError("SWGLoadImageBatch: no filenames given")

        input_dir = folder_paths.get_input_directory()
        tensors = []
        ref_size = None
        for name in names:
            path = os.path.join(input_dir, name)
            if not os.path.isfile(path):
                raise FileNotFoundError(f"SWGLoadImageBatch: missing input file {path}")
            img = Image.open(path)
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            if ref_size is None:
                ref_size = img.size
            elif img.size != ref_size:
                raise ValueError(
                    f"SWGLoadImageBatch: size mismatch, {name} is {img.size}, "
                    f"expected {ref_size} (a batch must be same-size images)"
                )
            arr = np.array(img).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr))

        batch = torch.stack(tensors, dim=0)
        return (batch,)

    @classmethod
    def IS_CHANGED(cls, filenames):
        # Always re-read from disk - staging files can change between runs
        # and correctness here matters more than the (tiny) cache savings.
        return time.time()


class SWGSaveImageBatch:
    """Save a batch tensor to disk with one filename per slot, in order.
    filenames must line up 1:1 with the batch that was fed into
    SWGLoadImageBatch earlier in the same graph, so the caller can compute
    expected output paths itself instead of parsing ComfyUI's own
    prefix+counter output metadata.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "filenames": ("STRING", {"multiline": True, "default": ""}),
                "subfolder": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_batch"
    OUTPUT_NODE = True
    CATEGORY = "swg_hd"

    def save_batch(self, images, filenames, subfolder):
        names = [n.strip() for n in filenames.splitlines() if n.strip()]
        if len(names) != images.shape[0]:
            raise ValueError(
                f"SWGSaveImageBatch: {images.shape[0]} images but {len(names)} "
                f"filenames given - must match 1:1 in order"
            )

        output_dir = folder_paths.get_output_directory()
        out_dir = os.path.join(output_dir, subfolder) if subfolder else output_dir
        os.makedirs(out_dir, exist_ok=True)

        for image, name in zip(images, names):
            arr = 255.0 * image.cpu().numpy()
            img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            img.save(os.path.join(out_dir, name))

        return {}

    @classmethod
    def IS_CHANGED(cls, images, filenames, subfolder):
        return time.time()


NODE_CLASS_MAPPINGS = {
    "SWGLoadImageBatch": SWGLoadImageBatch,
    "SWGSaveImageBatch": SWGSaveImageBatch,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SWGLoadImageBatch": "SWG Load Image Batch",
    "SWGSaveImageBatch": "SWG Save Image Batch",
}
