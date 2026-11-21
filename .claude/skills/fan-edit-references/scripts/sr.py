"""Real-ESRGAN general x4v3 (SRVGGNetCompact) on CPU, with denoise-strength blending (DNI)."""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import os
WORK = os.environ.get('FAN_EDIT_WORK', os.path.expanduser('~/fan-edits'))   # work folder: sources, models, outputs

M = WORK + '/models/'


class SRVGGNetCompact(nn.Module):
    def __init__(self, num_feat=64, num_conv=32, upscale=4):
        super().__init__()
        self.upscale = upscale
        self.body = nn.ModuleList([nn.Conv2d(3, num_feat, 3, 1, 1), nn.PReLU(num_parameters=num_feat)])
        for _ in range(num_conv):
            self.body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            self.body.append(nn.PReLU(num_parameters=num_feat))
        self.body.append(nn.Conv2d(num_feat, 3 * upscale * upscale, 3, 1, 1))
        self.upsampler = nn.PixelShuffle(upscale)

    def forward(self, x):
        out = x
        for layer in self.body: out = layer(out)
        out = self.upsampler(out)
        return out + F.interpolate(x, scale_factor=self.upscale, mode='nearest')


_net = None


def net(denoise=0.3):
    global _net
    if _net is None:
        a = torch.load(M + 'realesr-general-x4v3.pth', map_location='cpu', weights_only=True)['params']
        b = torch.load(M + 'realesr-general-wdn-x4v3.pth', map_location='cpu', weights_only=True)['params']
        sd = {k: (1 - denoise) * a[k] + denoise * b[k] for k in a}   # a = sharp, b = denoised
        _net = SRVGGNetCompact(); _net.load_state_dict(sd); _net.eval()
        torch.set_num_threads(4)
    return _net


@torch.inference_mode()
def upscale(img):
    """img: HxWx3 uint8 RGB -> (4H)x(4W)x3 uint8"""
    x = torch.from_numpy(img).permute(2, 0, 1)[None].float() / 255
    y = net()(x).clamp_(0, 1)[0].permute(1, 2, 0).mul_(255).round_().byte().numpy()
    return y
