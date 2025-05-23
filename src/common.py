# test/common.py

import time
import torch
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def get_sr_model(model_path: Path = "./src/models/pretrained_models/RealESRGAN_x2plus.pth", is_2x: bool = True):
    
    if is_2x:
        from .rrdb_net import RRDBNet        
    else:
        from basicsr.archs.rrdbnet_arch import RRDBNet
        
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
    load_net = torch.load(model_path)
    model.load_state_dict(load_net['params_ema'], strict=True)
    model.eval()
    model = model.to('cuda').half()
    return model

def show(x: torch.Tensor):
    print(x.shape)
    x = x[0].clip(0, 1).cpu().detach().numpy().transpose(1, 2, 0).astype(np.float32)
    plt.imshow(x)
    plt.show()


@torch.no_grad()
def benchmark(model, x, warm_up=2, runs=10):
    print("Warm up ...")
    with torch.no_grad():
        for _ in range(warm_up):
            features = model(x)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, runs + 1):
            start_time = time.time()
            features = model(x)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 10 == 0:
                print('Iteration %d/%d, ave batch time %.2f ms' % (i, runs, np.mean(timings[i-10: i]) * 1000))

    print("Input shape:", x.shape)
    print("Output features size:", features.shape)
    print('Average batch time: %.2f ms' % (np.mean(timings) * 1000))


def test(x, model, name):
    torch.cuda.empty_cache()
    s = time.time()
    with torch.no_grad():
        y = model(x)
    torch.cuda.synchronize()
    print(name, time.time() - s)
    show(y)
    return y


def prepare(side, model_path, is_2x):
    # Initialize the model with 3 input channels
    model = get_sr_model(model_path, is_2x).eval()
    
    # Load and preprocess the image
    x = cv2.resize(cv2.imread('./src/samples/sample.png'),
                   (side[0], side[1]))[..., ::-1].transpose(2, 0, 1) / 255.0
    x = torch.from_numpy(x).cuda().half().unsqueeze(0)
    
    if is_2x:
        b, c, h, w = x.size()
        h = h//2
        w = w//2
        x = x.view(b, c, h, 2, w, 2).permute(0, 1, 3, 5, 2, 4).reshape(b, 12, h, w)
    
    return model, x


