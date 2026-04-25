import torch
import numpy as np
from PIL import Image
import open_clip


class CLIPEmbedder:
    def __init__(self):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            'ViT-B-32', pretrained='openai'
        )
        self.tokenizer = open_clip.get_tokenizer('ViT-B-32')
        self.model.eval()

    def encode_image(self, image_path):
        image = self.preprocess(Image.open(image_path)).unsqueeze(0)

        with torch.no_grad():
            features = self.model.encode_image(image)

        emb = features[0].cpu().numpy()
        emb = emb / np.linalg.norm(emb)
        return emb

    def encode_text(self, text):
        tokens = self.tokenizer([text])

        with torch.no_grad():
            features = self.model.encode_text(tokens)

        emb = features[0].cpu().numpy()
        emb = emb / np.linalg.norm(emb)
        return emb