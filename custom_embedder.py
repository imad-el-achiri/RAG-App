from typing import List
from langchain_core.embeddings import Embeddings

import torch
from PIL import Image
import open_clip
torch.classes.__path__ = []
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClipEmbeddings(Embeddings):
    def __init__(self, model:str, pretrained:str):
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model, pretrained=pretrained, device=device)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = open_clip.get_tokenizer(model)
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        texts_embedded = self.tokenizer(texts)
        with torch.no_grad(), torch.autocast("cuda"):
            text_features = self.model.encode_text(texts_embedded.to(device))
            text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        return self.embed_documents([text])[0]

    def embed_image(self, uris:List[str]) -> List[List[float]]:
        images_processed = []
        for image in uris:
            images_processed.append(self.preprocess(Image.open(image)).unsqueeze(0).to(device))
        images_embeddings = []
        with torch.no_grad(), torch.autocast("cuda"):
            for image_processed in images_processed:
                image_features = self.model.encode_image(image_processed)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                images_embeddings.append(image_features[0].tolist())        
        return images_embeddings

    #def embed_image(self, image:str) -> List[List[float]]:
    #    return self.embed_images([image])