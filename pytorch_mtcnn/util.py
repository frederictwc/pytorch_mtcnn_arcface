import time
import math
import torch
USE_TRT = True
try:
    from torch2trt import TRTModule
except ModuleNotFoundError:
    print("torch2trt is not installed")
    USE_TRT = False


class Timer(object):
    def __init__(self, template_string):
        self.start = None
        self.template_string = template_string

    def __enter__(self):
        print("Start!")
        self.start = time.time()
        return self.start

    def __exit__(self, exc_type, exc_value, traceback):
        print("End")
        print(self.template_string.format(time.time() - self.start))
        return True


def get_image_pyramid_sizes(input_dimension, bottom_line):
    # BUILD AN IMAGE PYRAMID
    w, h = input_dimension
    shorter = min(h, w)
    min_detection_size = 12
    factor = 0.707  # sqrt(0.5)
    scales = []
    m = min_detection_size / bottom_line
    shorter *= m
    factor_count = 0
    while shorter > min_detection_size:
        scales.append(m * factor ** factor_count)
        shorter *= factor
        factor_count += 1
    pyramid_size = [(math.ceil(w * scale), math.ceil(h * scale)) for scale in scales]
    return scales, pyramid_size


if USE_TRT is True:
    def load_trt_model(path):
        model_trt = TRTModule()
        model_trt.load_state_dict(torch.load(path))
        return model_trt
