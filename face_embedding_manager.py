import os
from os.path import join, isdir, isfile
import shutil
from functools import reduce
import torch
from PIL import Image
from .pytorch_mtcnn.inference import DetectorInference, DetectorInferenceRT
from .pytorch_arcface.inference import EmbeddingExtractor


class FaceEmbeddingManager:
    def __init__(self, limit=20, min_face_size=30, thresholds=(0.2, 0.5, 0.8), nms_thresholds=(0.7, 0.7, 0.7),
                 facebank_directory='pytorch_mtcnn_arcface/facebank', mode='worker', use_tensorrt=False, rt_input_size=None):
        self._facebank_directory = facebank_directory
        self.__mode = mode
        self.thresholds = thresholds
        self.nms_thresholds = nms_thresholds
        self.limit = limit
        self.min_face_size = min_face_size
        self._use_tensorrt = use_tensorrt
        if use_tensorrt and rt_input_size is not None:
            self.detector = DetectorInferenceRT(
                'pytorch_mtcnn_arcface/pytorch_mtcnn/mtcnn_pytorch/src/weights/',
                rt_input_size,
                min_face_size
            )
        else:
            self.detector = DetectorInference(
                'pytorch_mtcnn_arcface/pytorch_mtcnn/mtcnn_pytorch/src/weights/pnet.npy',
                'pytorch_mtcnn_arcface/pytorch_mtcnn/mtcnn_pytorch/src/weights/rnet.npy',
                'pytorch_mtcnn_arcface/pytorch_mtcnn/mtcnn_pytorch/src/weights/onet.npy',
            )
        self.embedding_extractor = EmbeddingExtractor('pytorch_mtcnn_arcface/pytorch_arcface/checkpoints/model_ir_se50.pth')
        if isfile(join(self._facebank_directory, 'facebank.pth')):
            pass
        else:
            print("Facebank embedding not Found")
            os.makedirs(self._facebank_directory, exist_ok=True)
            self._save_facebank({'embeddings': torch.tensor([]), 'identities': list()})
            # if mode == 'stand_alone':
            #     print("Create facebank embedding from facebank folder")
            #     if not self._process_facebank_directory():
            #         raise SystemExit
            #     facebank = self._initiate_embeddings_and_identities()
            # else:
            #     raise SystemExit
            # self._save_facebank(facebank)
            # for content in os.scandir(self._facebank_directory):
            #     try:
            #         shutil.rmtree(content.path)
            #     except NotADirectoryError:
            #         pass

    def crop_and_align_images(self, images):
        """
        Crop and align the given images.
        :param images: List of PIL.Image of length M
        :return: List of Tuple of (numpay.ndarray of shape (N, 5) , list of (PIL.Image) of length N)
        """
        if self._use_tensorrt:
            result = [
                self.detector(x, limit=self.limit, thresholds=self.thresholds,
                              nms_thresholds=self.nms_thresholds) for x in images]
        else:
            result = [
                self.detector(x, limit=self.limit, min_face_size=self.min_face_size,
                              thresholds=self.thresholds, nms_thresholds=self.nms_thresholds) for x in images]
        return result

    def _process_facebank_directory(self):
        """
        Process the images under facebank directory
        :return: True on success, else False
        """
        folders = [x for x in os.scandir(self._facebank_directory) if x.is_dir()]
        original_images = {
            folder.name: [x for x in os.scandir(folder) if x.is_file() and (
                x.name.endswith('jpg') or x.name.endswith('png') or x.name.endswith('jpeg') or x.name.endswith('JPG')
            )] for folder in folders
        }
        empty_folder = [key for key, value in original_images.items() if len(value) == 0]
        for e in empty_folder:
            del original_images[e]
        processed_images = dict()
        for folder, images in original_images.items():
            pils = [Image.open(x.path) for x in images]
            cropped_faces = self.crop_and_align_images(pils)
            if not cropped_faces:
                print(f"No face is found in {folder} folder")
                break
            else:
                processed_images.update({folder: cropped_faces})
        else:
            for folder, images in original_images.items():
                for image in images:
                    os.remove(image.path)
            for folder, images in processed_images.items():
                for idx, image in enumerate(images):
                    image.save(join(self._facebank_directory, folder, '{idx}.jpg'))
            return True
        return False

    def _initiate_embeddings_and_identities(self):
        """
        Create facebank dictionary, {'embeddings': Tensor, 'identities': List of str}
        :return: Facebank dictionary
        """
        folders = [x for x in os.scandir(self._facebank_directory) if x.is_dir()]
        identities = list()
        embeddings = list()
        for folder in folders:
            files = [x for x in os.scandir(folder) if x.is_file()]
            files = [x for x in files if x.name.endswith('jpg') or x.name.endswith('png') or x.name.endswith('jpeg') or x.name.endswith('JPG')]
            if files:
                identities.append(folder.name)
                images = [Image.open(file.path) for file in files]
                embeddings.append(self.embedding_extractor(images).mean(dim=0).unsqueeze(0))
        embeddings = torch.cat(embeddings, dim=0)
        return {'embeddings': embeddings, 'identities': identities}

    def _get_facebank(self):
        """
        Load facebank from file system
        :return:
        """
        return torch.load(join(self._facebank_directory, 'facebank.pth'), map_location=torch.device('cpu'))

    def _save_facebank(self, facebank):
        """
        Save facebank to file system
        :param facebank: Facebank to save
        :return:
        """
        torch.save(facebank, join(self._facebank_directory, 'facebank.pth'))

    def recognize(self, embeddings, similarity_func='l2', l2_threshold=1.3, cosine_threshold=0.3):
        """
        Compare embeddings to given facebank
        :param embeddings: Tensor of shape [N, 512], where N is the number of faces in the input image
        :param similarity_func: Similarity function to be used, either one of 'l2' or 'cosine'
        :param l2_threshold: Threshold of l2, default to be 1.5
        :param cosine_threshold: Threshold of cosine, default to be 0.3
        :return: list of length N of identity index, list of length N of distance
        """
        facebank = self._get_facebank()
        facebank_embedding = facebank['embeddings']
        identities = facebank['identities']
        if similarity_func == 'l2':
            diff = embeddings.unsqueeze(-1) - facebank_embedding.transpose(1, 0).unsqueeze(0)
            dist = torch.sum(torch.pow(diff, 2), dim=1)
            face_distance, face_idx = torch.min(dist, dim=1)
            face_idx[face_distance > l2_threshold] = -1

        elif similarity_func == 'cosine':
            cosine_distance = torch.nn.modules.distance.CosineSimilarity(dim=1)
            dist = cosine_distance(embeddings, facebank_embedding)
            face_distance, face_idx = torch.max(dist, dim=0)
            face_idx[face_distance < cosine_threshold] = -1
        else:
            raise ValueError(f"Similarity Function {similarity_func} not implemented")
        return ['Unknown' if x == -1 else identities[x] for x in face_idx.tolist()], face_distance.tolist()

    def update_facebank(self, images, identity):
        """
        Updating existing facebank
        :param images: List of pillow images which are not cropped and aligned
        :param identity: the identity of the images
        :return: True on success, else False
        """
        if self.__mode == 'worker':
            raise NotImplementedError("Worker Mode does not support registration")
        facebank = self._get_facebank()
        facebank_embedding = facebank['embeddings']
        identities = facebank['identities']
        bboxes_list, cropped_faces_list = zip(*self.crop_and_align_images(images))
        cropped_faces = reduce(lambda x, y: x + y, cropped_faces_list)
        if not cropped_faces:
            raise ValueError('No face is found in input images')
        embedding = self.embedding_extractor(cropped_faces).mean(dim=0).unsqueeze(0)
        try:
            index = identities.index(identity)
            facebank_embedding[index] = embedding
        except ValueError:
            print("User does not exist")
            print("Register as a new user")
            identities.append(identity)

            facebank_embedding = torch.cat([facebank_embedding, embedding], dim=0)
        finally:
            self._save_facebank({'embeddings': facebank_embedding, 'identities': identities})

    def remove_identity(self, identity):
        """
        Remove a user from facebank
        :param identity:
        :return:
        """
        if self.__mode == 'worker':
            raise NotImplementedError("Worker Mode does not support deletion")
        facebank = self._get_facebank()
        facebank_embedding = facebank['embeddings']
        identities = facebank['identities']
        try:
            index = identities.index(identity)
            identities.pop(index)
            facebank_embedding = facebank_embedding.index_select(
                0, torch.tensor([x for x in range(facebank_embedding.shape[0]) if x != index])
            )
        except ValueError:
            print(f"Identity {identity} not found in facebank")
        finally:
            self._save_facebank({'embeddings': facebank_embedding, 'identities': identities})

    def get_existing_identities(self):
        facebank = self._get_facebank()
        identities = facebank['identities']
        return identities
