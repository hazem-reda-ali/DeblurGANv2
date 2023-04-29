import numpy as np
import torch.nn as nn
from skimage.metrics import structural_similarity as SSIM
from util.metrics import PSNR


class DeblurModel(nn.Module):
    def __init__(self):
        super(DeblurModel, self).__init__()

    def get_input(self, data):
        img = data['a']
        inputs = img
        targets = data['b']
        inputs, targets = inputs.cuda(), targets.cuda()
        return inputs, targets

    def tensor2im(self, image_tensor, imtype=np.uint8):
        image_numpy = image_tensor[0].cpu().float().numpy()
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
        return image_numpy.astype(imtype)

    def get_images_and_metrics(self, inp, output, target) -> (float, float, np.ndarray):
        inp = self.tensor2im(inp)
        fake = self.tensor2im(output.data)
        real = self.tensor2im(target.data)
        psnr = PSNR(fake, real)
        #ssim = SSIM(fake, real, multichannel=True) #line commented to run winsize auto set

        image_size = min(fake.shape[-2:]) # Get the smaller image dimension                                                     #New auto winsize        
        win_size = min(7, image_size) if image_size > 1 else 1 # Set win_size to 7 or smaller dimension, but at least 1         #New auto winsize  
        ssim = SSIM(fake, real, win_size=win_size, multichannel=True)                                                           #New auto winsize  


        vis_img = np.hstack((inp, fake, real))
        return psnr, ssim, vis_img


def get_model(model_config):
    return DeblurModel()
