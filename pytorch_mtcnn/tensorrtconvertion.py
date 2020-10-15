import torch
from torch2trt import torch2trt
from .mtcnn_pytorch.src.get_nets import PNet, RNet, ONet
import math
import time
from .util import get_image_pyramid_sizes
import argparse


parser = argparse.ArgumentParser("Optimised MTCNN with TensorRT")
parser.add_argument("--height", type=int, required=True)
parser.add_argument("--width", type=int, required=True)
parser.add_argument("--minimum", type=int, required=True)
args = parser.parse_args()


INPUT_IMAGE_SIZE = (args.width, args.height)
MIN_FACE_SIZE = args.minimum


# BUILD AN IMAGE PYRAMID
_, pyramid_sizes = get_image_pyramid_sizes(INPUT_IMAGE_SIZE, MIN_FACE_SIZE)


p_net_path = 'pytorch_mtcnn/mtcnn_pytorch/src/weights/pnet.npy'
r_net_path = 'pytorch_mtcnn/mtcnn_pytorch/src/weights/rnet.npy'
o_net_path = 'pytorch_mtcnn/mtcnn_pytorch/src/weights/onet.npy'


# create some regular pytorch model...
pnet = PNet().eval().cuda().half()
rnet = RNet().eval().cuda().half()
onet = ONet().eval().cuda().half()

# load weight
pnet.load_weight(p_net_path)
rnet.load_weight(r_net_path)
onet.load_weight(o_net_path)

# create example data
pnet_inputs = [torch.rand((1, 3, sh, sw)).cuda().half() for sw, sh in pyramid_sizes]
rnet_input = torch.rand((1, 3, 24, 24)).cuda().half()
onet_input = torch.rand((1, 3, 48, 48)).cuda().half()

# convert to TensorRT feeding sample data as input
pnet_trts = [torch2trt(pnet, [pnet_input], fp16_mode=True) for pnet_input in pnet_inputs]
rnet_trt = torch2trt(rnet, [rnet_input], max_batch_size=1024, fp16_mode=True)
onet_trt = torch2trt(onet, [onet_input], max_batch_size=1024, fp16_mode=True)

# warm up
[pnet(x) for x in pnet_inputs]
rnet(rnet_input)
onet(onet_input)


for i, pnet_trt in enumerate(pnet_trts):
    torch.save(pnet_trt.state_dict(), f'pnet_trt_{INPUT_IMAGE_SIZE[0]}_{INPUT_IMAGE_SIZE[1]}_{MIN_FACE_SIZE}_{i}.pth')
torch.save(rnet_trt.state_dict(), 'rnet_trt.pth')
torch.save(onet_trt.state_dict(), 'onet_trt.pth')
