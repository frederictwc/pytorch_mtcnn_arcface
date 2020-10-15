from torch2trt import torch2trt, TRTModule
from .mtcnn_pytorch.src.get_nets import PNet, RNet, ONet
import argparse
from .util import get_image_pyramid_sizes, Timer, load_trt_model
import torch


parser = argparse.ArgumentParser("Performance measure for TensorRT Optimised MTCNN")
parser.add_argument("--height", type=int, required=True)
parser.add_argument("--width", type=int, required=True)
parser.add_argument("--minimum", type=int, required=True)
args = parser.parse_args()


INPUT_IMAGE_SIZE = (args.width, args.height)
MIN_FACE_SIZE = args.minimum

_, pyramids = get_image_pyramid_sizes(INPUT_IMAGE_SIZE, MIN_FACE_SIZE)

p_net_path = 'pytorch_mtcnn/mtcnn_pytorch/src/weights/pnet.npy'
r_net_path = 'pytorch_mtcnn/mtcnn_pytorch/src/weights/rnet.npy'
o_net_path = 'pytorch_mtcnn/mtcnn_pytorch/src/weights/onet.npy'

# create some regular pytorch model...
pnet = PNet(p_net_path).eval().cuda()
rnet = RNet(r_net_path).eval().cuda()
onet = ONet(o_net_path).eval().cuda()


# convert to TensorRT feeding sample data as input
pnet_trts = [load_trt_model(f'pnet_trt_{INPUT_IMAGE_SIZE[0]}_{INPUT_IMAGE_SIZE[1]}_{MIN_FACE_SIZE}_{i}.pth') for i, _ in enumerate(pyramids)]
rnet_trt = load_trt_model('rnet_trt.pth')
onet_trt = load_trt_model('onet_trt.pth')

# create example data
pnet_inputs = [torch.rand((1, 3, sh, sw)).cuda() for sw, sh in pyramids]
rnet_input = torch.rand((1, 3, 24, 24)).cuda()
onet_input = torch.rand((1, 3, 48, 48)).cuda()


pnet_inputs_half = [i.half() for i in pnet_inputs]
rnet_input_half = rnet_input.half()
onet_input_half = onet_input.half()

# warm up
[pnet(x) for x in pnet_inputs]
rnet(rnet_input)
onet(onet_input)
[pnet_trts[i](pnet_inputs_half[i]) for i in range(len(pnet_trts))]
rnet_trt(rnet_input_half)
onet_trt(onet_input_half)

# time and calculate difference
with Timer("Original Pnet Inference Time: {}") as _:
    pnet_outputs = [pnet(x) for x in pnet_inputs]

with Timer("TRT Pnet Inference Time: {}") as _:
    pnet_trt_outputs = [pnet_trts[i](pnet_inputs_half[i]) for i in range(len(pnet_trts))]

with Timer("Original RNet Inference Time: {}") as _:
    rnet_output = rnet(rnet_input)

with Timer("TRT RNet Inference Time: {}") as _:
    rnet_trt_output = rnet_trt(rnet_input_half)

with Timer("Original ONet Inference Time: {}") as _:
    onet_output = onet(onet_input)

with Timer("TRT Onet Inference Time: {}") as _:
    onet_trt_output = onet_trt(onet_input_half)

# calcuate difference
for pnet_output, pnet_trt_output in zip(pnet_outputs, pnet_trt_outputs):
    print(torch.max(torch.abs(pnet_output[0] - pnet_trt_output[0])))
    print(torch.max(torch.abs(pnet_output[1] - pnet_trt_output[1])))
print(torch.max(torch.abs(rnet_output[0] - rnet_trt_output[0])))
print(torch.max(torch.abs(rnet_output[1] - rnet_trt_output[1])))
print(torch.max(torch.abs(onet_output[0] - onet_trt_output[0])))
print(torch.max(torch.abs(onet_output[1] - onet_trt_output[1])))
