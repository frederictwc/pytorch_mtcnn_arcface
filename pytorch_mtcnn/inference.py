import os
import numpy as np
import torch
from PIL import Image
import cv2
from .mtcnn_pytorch.src.get_nets import PNet, RNet, ONet
from .mtcnn_pytorch.src.box_utils import nms, calibrate_box, get_image_boxes_v2, convert_to_square, _preprocess
from torchvision.ops import nms as torchnms
from .mtcnn_pytorch.src.first_stage import run_first_stage, _generate_bboxes
from .mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
from .util import get_image_pyramid_sizes, USE_TRT
if USE_TRT:
    from .util import load_trt_model


class DetectorInference:
    def __init__(
            self, p_net_path, r_net_path, o_net_path):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.pnet = PNet(p_net_path).to(self.device)
        self.rnet = RNet(r_net_path).to(self.device)
        self.onet = ONet(o_net_path).to(self.device)
        self.pnet.eval()
        self.rnet.eval()
        self.onet.eval()
        self.refrence = get_reference_facial_points(default_square=True)

    def detect_faces(self, image, min_face_size=50.0,
                     thresholds=(0.6, 0.7, 0.8), nms_thresholds=(0.5, 0.7, 0.7)):
        """
        Arguments:
            image: an instance of numpy.array of uint8 obtained from np.array(PIL.Image).
            min_face_size: minimum pixel size of a face
            thresholds: tuple of three stages threshold
            nms_thresholds: tuple of three stages nms-threshold
        Returns:
            two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
            bounding boxes and facial landmarks.
        """

        # BUILD AN IMAGE PYRAMID
        height, width = image.shape[:2]
        min_length = min(height, width)
        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)
        scales = []
        m = min_detection_size / min_face_size
        min_length *= m

        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m*factor**factor_count)
            min_length *= factor
            factor_count += 1

        # -------------------------STAGE 1---------------------------------------------------

        # it will be returned
        bounding_boxes = []

        with torch.no_grad():
            # run P-Net on different scales
            for s in scales:
                boxes = run_first_stage(image, self.pnet, scale=s, threshold=thresholds[0])
                bounding_boxes.append(boxes)

            # collect boxes (and offsets, and scores) from different scales
            bounding_boxes = [i for i in bounding_boxes if i is not None]
            if len(bounding_boxes) == 0:
                return [], []
            bounding_boxes = np.vstack(bounding_boxes)

            # keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
            keep = torchnms(torch.as_tensor(bounding_boxes[:, :4]), torch.as_tensor(bounding_boxes[:, 4:]), nms_thresholds[0]).numpy()
            bounding_boxes = bounding_boxes[keep]

            # use offsets predicted by pnet to transform bounding boxes
            bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
            # shape [n_boxes, 5]

            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            # ------------------------------------STAGE 2------------------------------------

            img_boxes = get_image_boxes_v2(bounding_boxes, image, size=24)
            if len(img_boxes) == 0:
                return [], []
            img_boxes = torch.as_tensor(img_boxes).to(self.device)

            output = self.rnet(img_boxes)
            offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > thresholds[1])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]

            # keep = nms(bounding_boxes, nms_thresholds[1])
            keep = torchnms(torch.as_tensor(bounding_boxes[:, :4]), torch.as_tensor(bounding_boxes[:, 4:]), nms_thresholds[1]).numpy()
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
            bounding_boxes = convert_to_square(bounding_boxes)
            bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

            # -----------------------------------STAGE 3-------------------------------------

            img_boxes = get_image_boxes_v2(bounding_boxes, image, size=48)
            if len(img_boxes) == 0: 
                return [], []
            img_boxes = torch.as_tensor(img_boxes).to(self.device)
            output = self.onet(img_boxes)
            landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
            offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
            probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]

            keep = np.where(probs[:, 1] > thresholds[2])[0]
            bounding_boxes = bounding_boxes[keep]
            bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
            offsets = offsets[keep]
            landmarks = landmarks[keep]

            # compute landmark points
            width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
            height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
            xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
            landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
            landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

            bounding_boxes = calibrate_box(bounding_boxes, offsets)
            # keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
            keep = torchnms(torch.as_tensor(bounding_boxes[:, :4]), torch.as_tensor(bounding_boxes[:, 4:]), nms_thresholds[2]).numpy()
            bounding_boxes = bounding_boxes[keep]
            landmarks = landmarks[keep]
        return bounding_boxes, landmarks

    def __call__(self, img, limit=20, min_face_size=50.0,
                 thresholds=(0.1, 0.9, 0.9), nms_thresholds=(0.7, 0.7, 0.7)):
        """
        Return bboxes and cropped faces of an image, if no one is found in the image, two empty list will be returned
        Check length to for determine whether it is empty
        :param img Pillow Image
        :param limit Int
        :param min_face_size Float
        :param thresholds Tuple of three float
        :param nms_param thresholds Tuple of three float
        """
        img_arr = np.array(img)
        boxes, landmarks = self.detect_faces(img_arr, min_face_size, thresholds, nms_thresholds)
        if limit:
            boxes = boxes[:limit]
            landmarks = landmarks[:limit]
        faces = []
        for landmark in landmarks:
            facial5points = [[landmark[j], landmark[j + 5]] for j in range(5)]
            warped_face = warp_and_crop_face(img_arr, facial5points, self.refrence, crop_size=(112, 112))
            faces.append(Image.fromarray(warped_face))
        return boxes, faces


if USE_TRT:
    class DetectorInferenceRT:
        def __init__(
                self, tensorrt_checkpoint_dir, input_size, min_face_size):
            self.device = torch.device("cuda")
            self.input_size = input_size
            self.min_face_size = min_face_size
            self.scales, self.pyramids = get_image_pyramid_sizes(self.input_size, self.min_face_size)
            self.pnet = [load_trt_model(os.path.join(tensorrt_checkpoint_dir, f'pnet_trt_{self.input_size[0]}_{self.input_size[1]}_{self.min_face_size}_{i}.pth')) for
                         i, _ in enumerate(self.pyramids)]
            self.rnet = load_trt_model(os.path.join(tensorrt_checkpoint_dir, 'rnet_trt.pth'))
            self.onet = load_trt_model(os.path.join(tensorrt_checkpoint_dir, 'onet_trt.pth'))
            self.refrence = get_reference_facial_points(default_square=True)

        def _run_first_stage(self, image, net, scale, pyramid, threshold):
            """Run P-Net, generate bounding boxes, and do NMS.

            Arguments:
                image: an instance of numpy.array of uint8 obtain from np.array(PIL.Image).
                net: an instance of pytorch's nn.Module, P-Net.
                scale: a float number,
                    scale width and height of the image by this number.
                threshold: a float number,
                    threshold on the probability of a face when generating
                    bounding boxes from predictions of the net.

            Returns:
                a float numpy array of shape [n_boxes, 9],
                    bounding boxes with scores and offsets (4 + 1 + 4).
            """

            # scale the image and convert it to a float array
            # img = image.resize(pyramid, Image.BILINEAR)
            img = cv2.resize(image, pyramid, cv2.INTER_LINEAR).astype(np.float32)
            img = torch.FloatTensor(_preprocess(img)).to(self.device).half()
            with torch.no_grad():
                output = net(img)
                probs = output[1].cpu().data.numpy()[0, 1, :, :]
                offsets = output[0].cpu().data.numpy()
                # probs: probability of a face at each sliding window
                # offsets: transformations to true bounding boxes

                boxes = _generate_bboxes(probs, offsets, scale, threshold)
                if len(boxes) == 0:
                    return None

                # keep = nms(boxes[:, 0:5], overlap_threshold=0.5)
                keep = torchnms(torch.as_tensor(boxes[:, :4]), torch.as_tensor(boxes[:, 4:]), 0.5).numpy()
            return boxes[keep]

        def detect_faces(self, image,
                         thresholds=(0.6, 0.7, 0.8), nms_thresholds=(0.5, 0.7, 0.7)):
            """
            Arguments:
                image: an instance of numpy.array of uint8 obtained from np.array(PIL.Image).
                min_face_size: minimum pixel size of a face
                thresholds: tuple of three stages threshold
                nms_thresholds: tuple of three stages nms-threshold
            Returns:
                two float numpy arrays of shapes [n_boxes, 4] and [n_boxes, 10],
                bounding boxes and facial landmarks.
            """

            # -------------------------STAGE 1---------------------------------------------------

            # it will be returned
            bounding_boxes = []

            with torch.no_grad():
                # run P-Net on different scales
                for i, s in enumerate(self.scales):
                    boxes = self._run_first_stage(image, self.pnet[i], scale=s, pyramid=self.pyramids[i], threshold=thresholds[0])
                    bounding_boxes.append(boxes)

                # collect boxes (and offsets, and scores) from different scales
                bounding_boxes = [i for i in bounding_boxes if i is not None]
                if len(bounding_boxes) == 0:
                    return [], []
                bounding_boxes = np.vstack(bounding_boxes)

                # keep = nms(bounding_boxes[:, 0:5], nms_thresholds[0])
                keep = torchnms(torch.as_tensor(bounding_boxes[:, :4]), torch.as_tensor(bounding_boxes[:, 4:]), nms_thresholds[0]).numpy()
                bounding_boxes = bounding_boxes[keep]

                # use offsets predicted by pnet to transform bounding boxes
                bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
                # shape [n_boxes, 5]

                bounding_boxes = convert_to_square(bounding_boxes)
                bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

                # unkonwn issue: x1 > x2 or y1 > y2
                # maybe due to floating point downcast?
                # need further investigate
                # below lines are workaround
                bounding_boxes = bounding_boxes[bounding_boxes[:, 0] < bounding_boxes[:, 2]]
                bounding_boxes = bounding_boxes[bounding_boxes[:, 1] < bounding_boxes[:, 3]]
                bounding_boxes = bounding_boxes[bounding_boxes[:, 0] < image.shape[1]]
                bounding_boxes = bounding_boxes[bounding_boxes[:, 2] > 0.0]
                bounding_boxes = bounding_boxes[bounding_boxes[:, 1] < image.shape[0]]
                bounding_boxes = bounding_boxes[bounding_boxes[:, 3] > 0.0]
                # ------------------------------------STAGE 2------------------------------------

                img_boxes = get_image_boxes_v2(bounding_boxes, image, size=24)
                if len(img_boxes) == 0:
                    return [], []
                img_boxes = torch.as_tensor(img_boxes).to(self.device).half()

                output = self.rnet(img_boxes)
                offsets = output[0].cpu().data.numpy()  # shape [n_boxes, 4]
                probs = output[1].cpu().data.numpy()  # shape [n_boxes, 2]

                keep = np.where(probs[:, 1] > thresholds[1])[0]
                bounding_boxes = bounding_boxes[keep]
                bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
                offsets = offsets[keep]

                # keep = nms(bounding_boxes, nms_thresholds[1])
                keep = torchnms(torch.as_tensor(bounding_boxes[:, :4]), torch.as_tensor(bounding_boxes[:, 4:]), nms_thresholds[1]).numpy()
                bounding_boxes = bounding_boxes[keep]
                bounding_boxes = calibrate_box(bounding_boxes, offsets[keep])
                bounding_boxes = convert_to_square(bounding_boxes)
                bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])
                # unkonwn issue: x1 > x2 or y1 > y2
                # maybe due to floating point downcast?
                # need further investigate
                # below lines are workaround
                bounding_boxes = bounding_boxes[bounding_boxes[:, 0] < bounding_boxes[:, 2]]
                bounding_boxes = bounding_boxes[bounding_boxes[:, 1] < bounding_boxes[:, 3]]
                bounding_boxes = bounding_boxes[bounding_boxes[:, 0] < image.shape[1]]
                bounding_boxes = bounding_boxes[bounding_boxes[:, 2] > 0.0]
                bounding_boxes = bounding_boxes[bounding_boxes[:, 1] < image.shape[0]]
                bounding_boxes = bounding_boxes[bounding_boxes[:, 3] > 0.0]
                # -----------------------------------STAGE 3-------------------------------------

                img_boxes = get_image_boxes_v2(bounding_boxes, image, size=48)
                if len(img_boxes) == 0:
                    return [], []
                img_boxes = torch.as_tensor(img_boxes).to(self.device).half()
                output = self.onet(img_boxes)
                landmarks = output[0].cpu().data.numpy()  # shape [n_boxes, 10]
                offsets = output[1].cpu().data.numpy()  # shape [n_boxes, 4]
                probs = output[2].cpu().data.numpy()  # shape [n_boxes, 2]

                keep = np.where(probs[:, 1] > thresholds[2])[0]
                bounding_boxes = bounding_boxes[keep]
                bounding_boxes[:, 4] = probs[keep, 1].reshape((-1,))
                offsets = offsets[keep]
                landmarks = landmarks[keep]

                # compute landmark points
                width = bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0
                height = bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0
                xmin, ymin = bounding_boxes[:, 0], bounding_boxes[:, 1]
                landmarks[:, 0:5] = np.expand_dims(xmin, 1) + np.expand_dims(width, 1)*landmarks[:, 0:5]
                landmarks[:, 5:10] = np.expand_dims(ymin, 1) + np.expand_dims(height, 1)*landmarks[:, 5:10]

                bounding_boxes = calibrate_box(bounding_boxes, offsets)
                # keep = nms(bounding_boxes, nms_thresholds[2], mode='min')
                keep = torchnms(torch.as_tensor(bounding_boxes[:, :4]), torch.as_tensor(bounding_boxes[:, 4:]), nms_thresholds[2]).numpy()
                bounding_boxes = bounding_boxes[keep]
                landmarks = landmarks[keep]
            return bounding_boxes, landmarks

        def __call__(self, img, limit=20,
                     thresholds=(0.1, 0.9, 0.9), nms_thresholds=(0.7, 0.7, 0.7)):
            """
            Return bboxes and cropped faces of an image, if no one is found in the image, two empty list will be returned
            Check length to for determine whether it is empty
            :param img Pillow Image
            :param limit Int
            :param min_face_size Float
            :param thresholds Tuple of three float
            :param nms_param thresholds Tuple of three float
            """
            img_arr = np.array(img)
            boxes, landmarks = self.detect_faces(img_arr, thresholds, nms_thresholds)
            if limit:
                boxes = boxes[:limit]
                landmarks = landmarks[:limit]
            faces = []
            for landmark in landmarks:
                facial5points = [[landmark[j], landmark[j + 5]] for j in range(5)]
                warped_face = warp_and_crop_face(img_arr, facial5points, self.refrence, crop_size=(112, 112))
                faces.append(Image.fromarray(warped_face))
            return boxes, faces


if __name__ == "__main__":
    from .util import Timer
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    pil = Image.open('office4.jpg')
    # pil = Image.open('/Users/powatsoi/Workspace/python_camera_calbration_with_opencv/undistorted_1569315699.jpg')
    # detector = DetectorInference(
    #     'pytorch_mtcnn/mtcnn_pytorch/src/weights/pnet.npy',
    #     'pytorch_mtcnn/mtcnn_pytorch/src/weights/rnet.npy',
    #     'pytorch_mtcnn/mtcnn_pytorch/src/weights/onet.npy'
    # )
    # print(detector(pil))
    # with Timer("Without TensorRT: {}") as _:
    #     for i in range(10):
    #         detector(pil)
    # del detector
    detector = DetectorInferenceRT('.', (615, 407), 50)
    print(detector(pil))
    with Timer("With TensorRT: {}") as _:
        for i in range(10):
            detector(pil)
    # del detector
    import cProfile, pstats, io
    # from pstats import SortKey

    pr = cProfile.Profile()
    pr.enable()
    detector(pil)
    pr.disable()
    s = io.StringIO()
    # sortby = SortKey.CUMULATIVE
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats()
    print(s.getvalue())
