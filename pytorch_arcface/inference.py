import torch
from torchvision import transforms as T
from .face_recognizer_arcface import Backbone


class EmbeddingExtractor:
    def __init__(self, path_to_checkpoint):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = Backbone(50, mode='ir_se')
        self.model.load_state_dict(torch.load(path_to_checkpoint, map_location=self.device))
        self.model.eval()
        self.model.to(self.device)
        self.transform = T.Compose([T.ToTensor(),
                                    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def __call__(self, images):
        """
        Get embedding for a set of images of a single identities
        :param images: list of PIL Image
        :return: embeddings tensor in cpu
        """
        image_tensor = torch.cat([self.transform(image).unsqueeze(0) for image in images], dim=0)
        image_tensor = image_tensor.to(self.device)
        embeddings = self.model(image_tensor).to(torch.device('cpu'))
        return embeddings
