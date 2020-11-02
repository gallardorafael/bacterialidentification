"""
This script augments the original DIBaS dataset, which consists
in:
* 32 classes (species of bacteria)
* Aprox. 660 examples of colonies (aprox. 20 examples per specie)
"""

import argparse
import os
from PIL import Image
from torchvision import transforms
import torch

# Arguments
parser = argparse.ArgumentParser(description='Options for the augmentation')
parser.add_argument('--input_dir', default='./')
parser.add_argument('--output_dir', default='./')
parser.add_argument('--size', default=256)
parser.add_argument('--operation', default=None)
args = parser.parse_args()

# Function to rotate all the images in a folder
def rotate_imgs(angles):
    try:
        os.mkdir(args.output_dir)
    except OSError as error:
        print(error)
    for filename in os.listdir(args.input_dir):
        print('Rotating:',os.path.join(args.input_dir, filename))
        image = Image.open(os.path.join(args.input_dir, filename))
        for theta in angles:
            rotated = image.rotate(angle=theta, resample=Image.BICUBIC)
            r_name = filename + '_rotated_' + str(theta)
            path = os.path.join(args.output_dir, r_name)
            print('Saving:',path)
            rotated.save(fp=path+'.tif')

# Function to resize all the images in a folder
def resize(shape):
    try:
        os.mkdir(args.output_dir)
    except OSError as error:
        print(error)
    transform = transforms.Compose([transforms.Resize(shape, interpolation=Image.LANCZOS)])
    for filename in os.listdir(args.input_dir):
        print('Resizing:',os.path.join(args.input_dir, filename))
        image = Image.open(os.path.join(args.input_dir, filename))
        resized = transform(image)
        r_name = filename + '_resized_'
        path = os.path.join(args.output_dir, r_name)
        print('Saving:',path)
        resized.save(fp=path+'.tif')

# Function to TenCrop all the images in a folder
def TenCrop(shape):
    try:
        os.mkdir(args.output_dir)
    except OSError as error:
        print(error)
    transform = transforms.Compose([transforms.TenCrop(shape),
                                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])
    toPILImage = transforms.Compose([transforms.ToPILImage()])
    for filename in os.listdir(args.input_dir):
        print('TenCropping:',os.path.join(args.input_dir, filename))
        image = Image.open(os.path.join(args.input_dir, filename))
        crops_t = transform(image) # 4 dimensions tensor
        crops_ts = torch.split(crops_t, 1, 0) # List of 3 dimension tensors
        for i, img_t in enumerate(crops_ts):
            new_img_t = torch.squeeze(img_t)
            img = toPILImage(new_img_t)
            r_name = filename + '_cropped_' + str(i)
            path = os.path.join(args.output_dir, r_name)
            print('Saving:',path)
            img.save(fp=path+'.tif')

# Function to TenCrop the full dataset
def TenCropFull(shape):
    # If the augmented folder does not exist
    try:
        os.mkdir(args.output_dir)
    except OSError as error:
        print(error)
    transform = transforms.Compose([transforms.TenCrop(shape),
                                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])
    toPILImage = transforms.Compose([transforms.ToPILImage()])
    for specie in os.listdir(args.input_dir):
        specie_path = os.path.join(args.input_dir, specie)
        out_specie_path = os.path.join(args.output_dir, specie)
        # If the augmented specie folder does not exist
        try:
            os.mkdir(out_specie_path)
        except OSError as error:
            print(error)
        for filename in os.listdir(specie_path):
            print('TenCropping:',os.path.join(specie_path, filename))
            image = Image.open(os.path.join(specie_path, filename))
            crops_t = transform(image) # 4 dimensions tensor
            crops_ts = torch.split(crops_t, 1, 0) # List of 3 dimension tensors
            for i, img_t in enumerate(crops_ts):
                new_img_t = torch.squeeze(img_t)
                img = toPILImage(new_img_t)
                r_name = filename + '_cropped_' + str(i)
                path = os.path.join(out_specie_path, r_name)
                print('Saving:',path+'.tif')
                img.save(fp=path+'.tif')

# Function to FiveCrop the full dataset
def FiveCropFull(shape):
    # If the augmented folder does not exist
    try:
        os.mkdir(args.output_dir)
    except OSError as error:
        print(error)
    transform = transforms.Compose([transforms.FiveCrop(shape),
                                    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops]))])
    toPILImage = transforms.Compose([transforms.ToPILImage()])
    for specie in os.listdir(args.input_dir):
        specie_path = os.path.join(args.input_dir, specie)
        out_specie_path = os.path.join(args.output_dir, specie)
        # If the augmented specie folder does not exist
        try:
            os.mkdir(out_specie_path)
        except OSError as error:
            print(error)
        for filename in os.listdir(specie_path):
            print('FiveCropping:',os.path.join(specie_path, filename))
            image = Image.open(os.path.join(specie_path, filename))
            crops_t = transform(image) # 4 dimensions tensor
            crops_ts = torch.split(crops_t, 1, 0) # List of 3 dimension tensors
            for i, img_t in enumerate(crops_ts):
                new_img_t = torch.squeeze(img_t)
                img = toPILImage(new_img_t)
                r_name = filename + '5_cropped_' + str(i)
                path = os.path.join(out_specie_path, r_name)
                print('Saving:',path+'.tif')
                img.save(fp=path+'.tif')

# Function to resize the full dataset
def resizeFull(shape):
    try:
        os.mkdir(args.output_dir)
    except OSError as error:
        print(error)
    transform = transforms.Compose([transforms.Resize(shape, interpolation=Image.LANCZOS)])
    for specie in os.listdir(args.input_dir):
        specie_path = os.path.join(args.input_dir, specie)
        out_specie_path = os.path.join(args.output_dir, specie)
        # If the augmented specie folder does not exist
        try:
            os.mkdir(out_specie_path)
        except OSError as error:
            print(error)
        for filename in os.listdir(specie_path):
            print('Resizing:',os.path.join(specie_path, filename))
            image = Image.open(os.path.join(specie_path, filename))
            resized = transform(image)
            r_name = filename + '_resized_'
            path = os.path.join(out_specie_path, r_name)
            print('Saving:',path+'.tif')
            resized.save(fp=path+'.tif')

# Function to rotate the full dataset
def rotateFull(angles):
    try:
        os.mkdir(args.output_dir)
    except OSError as error:
        print(error)
    for specie in os.listdir(args.input_dir):
        specie_path = os.path.join(args.input_dir, specie)
        out_specie_path = os.path.join(args.output_dir, specie)
        # If the augmented specie folder does not exist
        try:
            os.mkdir(out_specie_path)
        except OSError as error:
            print(error)
        for filename in os.listdir(specie_path):
            print('Rotating:', os.path.join(specie_path, filename))
            image = Image.open(os.path.join(specie_path, filename))
            for theta in angles:
                rotated = image.rotate(angle=theta, resample=Image.BICUBIC)
                r_name = filename + '_rotated_' + str(theta)
                path = os.path.join(out_specie_path, r_name)
                print('Saving:',path+'.tif')
                rotated.save(fp=path+'.tif')
def main():
    if args.operation == None:
        print("Should specify the operation to perform.")
    elif args.operation == 'TenCropFull':
        pass
    elif args.operation == 'resizeFull':
        size_in = int(args.size)
        resizeFull((size_in, size_in))
    elif args.operation == 'rotateFull':
        pass

if __name__ == "__main__":
    main()
