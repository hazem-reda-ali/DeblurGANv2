import os
import argparse
from tqdm import tqdm

import torch
import torchvision.transforms as transforms
from PIL import Image

def get_generator(model_type):
    if model_type == 'resnet':
        generator = ...
    elif model_type == 'unet':
        generator = ...
    else:
        raise ValueError('Invalid model_type: {}'.format(model_type))
    return generator

def tensor2im(tensor):
    tensor = tensor.squeeze(0).detach().cpu()
    tensor = tensor.mul(0.5).add(0.5)
    tensor = tensor.clamp(0, 1)
    tensor = tensor.permute(1, 2, 0).numpy()
    tensor = (tensor * 255).astype('uint8')
    return tensor

def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)

def main(args):
    # Create output directory
    mkdirs(args.output_dir)

    # Load generator
    try:
        generator = get_generator(args.generator_type)
        checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'))
        generator.load_state_dict(checkpoint['model'])
        generator.eval()
    except Exception as e:
        print(f"Error loading generator: {e}")
        return

    # Load transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Get files
    files = get_files(args.input_dir, args.output_dir)

    # Generate images
    for blur_path, sharp_path in tqdm(files):
        blur_img = Image.open(blur_path).convert('RGB')
        blur_tensor = transform(blur_img).unsqueeze(0)

        with torch.no_grad():
            sharp_tensor = generator(blur_tensor)

        sharp_img = tensor2im(sharp_tensor)
        sharp_pil = Image.fromarray(sharp_img)

        # Save result
        sharp_pil.save(sharp_path)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Deblur images')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Path to directory containing blurred images')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Path to directory to save deblurred images')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to generator model checkpoint')
    args = parser.parse_args()

    main(args)