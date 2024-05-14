import numpy as np
import json
import os
import sys
import time
import matlab.engine
from mosaicing_demosaicing_v2 import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils
from torchvision import models
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import torchattacks
from torchattacks.attack import Attack
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize

from image_transformer import ImageTransformer
from utils import *


# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

class Morie_attack(Attack):
    r"""
    Distance Measure : L_inf bound on sensor noise
    Arguments:
        model (nn.Module): Victim model to attack.
        steps (int): number of steps. (DEFAULT: 50)
        batch_size (int): batch size
        scale_factor (int): zoom in the images on the LCD. （DEFAULT: 3）

    Shape:
        - images: :math:`(N, C, H, W)` where `N = number of batches`, `C = number of channels`, `H = height` and `W = width`. It must have a range [0, 1].
        - labels: :math:`(N)` where each value :math:`y_i` is :math:`0 \leq y_i \leq` `number of labels`.
        - output: :math:`(N, C, H, W)`.
    """

    def __init__(self, model, img_h, img_w, noise_budget, scale_factor, batch_size=50, targeted=False):
        super(Morie_attack, self).__init__("Morie_attack", model)
        self.targeted = targeted
        self.img_w = img_w
        self.img_h = img_h
        self.scale_factor = scale_factor
        self.noise_budget = noise_budget
        self.mean = 0  # 假设高斯噪声均值是0
        self.stddev = noise_budget  # 噪声预算作为标准差
        self.batch_size = batch_size
        '''
        noise = np.zeros([batch_size, self.img_h * self.scale_factor * 3, self.img_w * self.scale_factor * 3])
        self.noise = torch.from_numpy(noise).to(self.device)
        self.noise.requires_grad = True
        self.adv_loss = nn.CrossEntropyLoss()
        '''

    def simulate_LCD_display(self, input_img):
        """ Simulate the display of raw images on LCD screen
        Input:
            original images (tensor): batch x height x width x channel
        Output:
            LCD images (tensor): batch x (height x scale_factor)  x (width x scale_factor) x channel
        """
        input_img = np.asarray(input_img.cpu().detach())
        batch_size, h, w, c = input_img.shape

        simulate_imgs = np.zeros((batch_size, h * 3, w * 3, 3), dtype=np.float32)
        red = np.repeat(input_img[:, :, :, 0], 3, axis=1)
        green = np.repeat(input_img[:, :, :, 1], 3, axis=1)
        blue = np.repeat(input_img[:, :, :, 2], 3, axis=1)

        for y in range(w):
            simulate_imgs[:, :, y * 3, 0] = red[:, :, y]
            simulate_imgs[:, :, y * 3 + 1, 1] = green[:, :, y]
            simulate_imgs[:, :, y * 3 + 2, 2] = blue[:, :, y]
        simulate_imgs = torch.from_numpy(simulate_imgs).to(self.device)

        return simulate_imgs

    def demosaic_and_denoise(self, input_img):
        """ Apply demosaicing to the images
        Input:
            images (tensor): batch x (height x scale_factor) x (width x scale_factor)
        Output:
            demosaicing images (tensor): batch x (height x scale_factor) x (width x scale_factor) x channel
        """
        demosaicing_imgs = demosaicing_CFA_Bayer_bilinear(input_img)
        return demosaicing_imgs

    def simulate_CFA(self, input_img):
        """ Simulate the raw reading of the camera sensor using bayer CFA
        Input:
            images (tensor): batch x (height x scale_factor) x (width x scale_factor) x channel
        Output:
            mosaicing images (tensor): batch x (height x scale_factor) x (width x scale_factor)
        """
        mosaicing_imgs = mosaicing_CFA_Bayer(input_img)
        return mosaicing_imgs

    def random_rotation_3(self, org_images, lcd_images, theta, phi, gamma):
        """ Simulate the 3D rotatation during the shooting
        Input:
            images (tensor): batch x height x width x channel
        Rotate angle:
            theta (int): (-20, 20) ---- rotation around the x-axis in degrees
            phi (int): (-20, 20) ---- rotation around the y-axis in degrees
            gamma (int): (-20, 20) ---- rotation around the z-axis in degrees
        Output:
            rotated original images (tensor): batch x height x width x channel
            rotated LCD images (tensor): batch x (height x scale_factor) x (width x scale_factor) x channel
        """
        rotate_images = np.zeros(org_images.size())
        rotate_lcd_images = np.zeros(lcd_images.size())

        for n, img in enumerate(org_images):
            # Use the specific rotation angles theta, phi, gamma
            Trans_org = ImageTransformer(img)
            _, _, _, rotate_img = Trans_org.rotate_along_axis(False, phi=phi, gamma=gamma, theta=theta)
            rotate_images[n, :] = rotate_img

            Trans_lcd = ImageTransformer(lcd_images[n])
            _, _, _, rotate_lcd_img = Trans_lcd.rotate_along_axis(False, phi=phi, gamma=gamma, theta=theta)
            rotate_lcd_images[n, :] = rotate_lcd_img

        # Convert numpy arrays to tensors
        rotate_images = torch.from_numpy(rotate_images).to(device)
        rotate_lcd_images = torch.from_numpy(rotate_lcd_images).to(device)

        return rotate_images, rotate_lcd_images

    def forward(self, org_imgs, org_labels, targeted_labels, theta=0.0, phi=0.0, gamma=0.0):
        r"""
        Overridden.
        """
        org_images = org_imgs.clone().detach().to(self.device)

        resize_before_lcd = F.interpolate(org_images, scale_factor=self.scale_factor, mode="bilinear")
        resize_before_lcd = resize_before_lcd.permute(0, 2, 3, 1)
        lcd_images = self.simulate_LCD_display(resize_before_lcd)

        temp_images = org_images.clone().detach().permute(0, 2, 3, 1)

        rotate_images, rotate_lcd_images = self.random_rotation_3(temp_images, lcd_images, theta, phi, gamma)
        rotate_lcd_images = rotate_lcd_images.to(self.device).detach()

        # save rotate_lcd_images as PNG in current folder
        rotate_lcd_images = rotate_lcd_images.permute(0, 3, 1, 2)
        rotate_lcd_images = rotate_lcd_images.float()
        rotate_lcd_images = torch.clamp(rotate_lcd_images, min=0, max=255).detach()
        img_rotate_lcd_images = rotate_lcd_images[0].detach().cpu().numpy()
        img_rotate_lcd_images = np.moveaxis(img_rotate_lcd_images, 0, 2)
        img_rotate_lcd_images_name = f"LCD子像素旋转变换后图片.PNG"
        img_rotate_lcd_images_path = img_rotate_lcd_images_name
        img_rotate_lcd_images_pil = Image.fromarray(img_rotate_lcd_images.astype(np.uint8))
        img_rotate_lcd_images_pil.save(img_rotate_lcd_images_path)

        # psf_img = rotate_lcd_images

        eng = matlab.engine.start_matlab()
        # eng.cd(r'myFolder', nargout=0)
        eng.moire_pattern_with_python(nargout=0)
        time.sleep(5)

        ######################################################
        psf_img_path = 'LCD子像素旋转_psf卷积后图像_zemax.PNG'
        psf_img = Image.open(psf_img_path).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        psf_img = transform(psf_img)
        psf_img *= 255.0
        psf_img = psf_img.to(dtype=torch.float64, device=self.device)
        # psf_img = psf_img.clone().detach().to(dtype=torch.float32, device=self.device)
        psf_img = psf_img.unsqueeze(0)
        psf_img = psf_img.permute(0, 2, 3, 1)
        ######################################################

        cfa_img = self.simulate_CFA(psf_img)
        # Simulating the attack without iterative adjustment
        # Get the size of input images
        batch_size, channels, height, width = psf_img.size()

        # Generate Gaussian noise, 您可以根据需要调整`mean`和`stddev`来控制噪声的强度
        noise = self.stddev * torch.randn(cfa_img.size()).to(self.device) + self.mean
        noise = torch.clamp(noise, min=-self.noise_budget, max=self.noise_budget)  # 限制噪声范围
        # 直接将高斯噪声添加到图像上（确保噪声总是在允许的范围内）
        cfa_img_noise = cfa_img + noise  # 添加噪声到图像上
        # cfa_img_noise = cfa_img

        demosaic_img = self.demosaic_and_denoise(cfa_img_noise)
        demosaic_img = demosaic_img.permute(0, 3, 1, 2)

        ## Adjust the brightness
        brighter_img = adjust_contrast_and_brightness(demosaic_img, beta=0)

        at_images = F.interpolate(brighter_img, [self.img_h, self.img_w], mode='bilinear')

        # Feed the modified images through the model to get the predictions
        at_images = at_images.float()

        at_images = torch.clamp(at_images, min=0, max=255).detach()

        # 返回包含有噪声的图像及其模型预测
        return resize_before_lcd, lcd_images, rotate_lcd_images, psf_img, at_images


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        input = input / 255.0
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_idx = json.load(open("./data/imagenet_class_index.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]

    transform = transforms.Compose([
        transforms.ToTensor(), ])

    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    model = nn.Sequential(
        norm_layer,
        models.inception_v3(pretrained=True)
    ).to(device)

    model = model.eval()

    ## Save the results of MA
    Save_results = 'True'
    if Save_results == 'True':
        savedir = './Results'
        Smpf_dir = os.path.join(savedir, 'moire')
        adv_dir = os.path.join(savedir, 'adv')
        rotate_dir = os.path.join(savedir, 'rotate')
        org_dir = os.path.join(savedir, 'org')
        dim_dir = os.path.join(savedir, 'dim')
        create_dir(adv_dir)
        create_dir(rotate_dir)
        create_dir(org_dir)
        create_dir(dim_dir)
        create_dir(Smpf_dir)

    ## deffault settings
    noise_budget = 0
    batch_size = 1
    epoch = 1  # int(1000 / batch_size)

    # 定义旋转角度
    theta_val = -7.0
    phi_val = -15.5
    gamma_val = 0.0
    original_size_h = 338
    original_size_w = 450

    normal_data = image_folder_custom_label(root='./data/dataset/incepv3_data_1000', transform=transform,
                                            idx2label=class2label)
    normal_loader = torch.utils.data.DataLoader(normal_data, batch_size=batch_size, shuffle=False)
    normal_iter = iter(normal_loader)

    print("-" * 70)
    print("Noise_budget = ", noise_budget)
    start = time.time()
    for batch in range(epoch):

        print("-" * 70)
        # org_imgs, org_labels = normal_iter.next()
        org_imgs, org_labels = next(normal_iter)
        org_imgs = org_imgs * 255.0

        # 实例化没有迭代步骤的Morire攻击
        attack = Morie_attack(model, noise_budget=noise_budget, img_w=original_size_w, img_h=original_size_h,
                              scale_factor=3, targeted=True,
                              batch_size=batch_size)

        # 进行攻击而不需要迭代
        resize_before_lcd, lcd_images, rotate_lcd_images, psf_img, at_images = attack(org_imgs, org_labels, None,
                                                                                      theta_val, phi_val,
                                                                                      gamma_val)  # 假设attack方法已更新，不再需要`targeted_labels`

        resize_before_lcd = resize_before_lcd.permute(0, 3, 1, 2)
        resize_before_lcd = resize_before_lcd.float()
        resize_before_lcd = torch.clamp(resize_before_lcd, min=0, max=255).detach()
        lcd_images = lcd_images.permute(0, 3, 1, 2)
        lcd_images = lcd_images.float()
        lcd_images = torch.clamp(lcd_images, min=0, max=255).detach()

        psf_img = psf_img.permute(0, 3, 1, 2)
        psf_img = psf_img.float()
        psf_img = torch.clamp(psf_img, min=0, max=255).detach()

        # at_images是攻击后生成的图片，现在需要将它们缩放回原始尺寸
        # 假设原始图片的大小存储在变量original_size中（例如：(224, 224)）
        original_size = (original_size_h, original_size_w)
        # 将攻击后的图片缩放回原始尺寸
        at_images_resized = F.interpolate(at_images, size=original_size, mode='bilinear', align_corners=False)

        # at_images_resized的大小为[N, C, H, W]
        # 选择绿色通道 (索引为1)
        green_channel = at_images_resized[:, 1, :, :]
        # 将绿色通道的值减少一半
        green_channel *= 1

        # save the pics
        for i in range(batch_size):

            if Save_results == 'True':
                img_at = at_images_resized[i].detach().cpu().numpy()
                img_at = np.moveaxis(img_at, 0, 2)  # Transpose the axis from CHW to HWC for image saving

                img_resize_before_lcd = resize_before_lcd[i].detach().cpu().numpy()
                img_resize_before_lcd = np.moveaxis(img_resize_before_lcd, 0, 2)

                img_lcd_images = lcd_images[i].detach().cpu().numpy()
                img_lcd_images = np.moveaxis(img_lcd_images, 0, 2)



                img_psf_img = psf_img[i].detach().cpu().numpy()
                img_psf_img = np.moveaxis(img_psf_img, 0, 2)

                # Construct the filename for the adversarial example image
                img_at_name = f"rotate[{theta_val:.2f},{phi_val:.2f},{gamma_val:.2f}]_{original_size_h}x{original_size_w}_noiseBudget={noise_budget}.JPEG"
                img_at_path = os.path.join(Smpf_dir, img_at_name)
                img_at_pil = Image.fromarray(img_at.astype(np.uint8))
                img_at_pil.save(img_at_path)
                print(f"Saved adversarial image as: {img_at_path}")

                img_resize_before_lcd_name = f"缩放图片_[{theta_val:.2f},{phi_val:.2f},{gamma_val:.2f}]_{original_size_h}x{original_size_w}_noiseBudget={noise_budget}.JPEG"
                img_resize_before_lcd_name_path = os.path.join(Smpf_dir, img_resize_before_lcd_name)
                img_resize_before_lcd_name_pil = Image.fromarray(img_resize_before_lcd.astype(np.uint8))
                img_resize_before_lcd_name_pil.save(img_resize_before_lcd_name_path)

                img_lcd_images_name = f"LCD子像素图片_[{theta_val:.2f},{phi_val:.2f},{gamma_val:.2f}]_{original_size_h}x{original_size_w}_noiseBudget={noise_budget}.JPEG"
                img_lcd_images_path = os.path.join(Smpf_dir, img_lcd_images_name)
                img_lcd_images_pil = Image.fromarray(img_lcd_images.astype(np.uint8))
                img_lcd_images_pil.save(img_lcd_images_path)



                img_psf_img_name = f"LCD子像素旋转变换_psf卷积后图片_[{theta_val:.2f},{phi_val:.2f},{gamma_val:.2f}]_{original_size_h}x{original_size_w}_noiseBudget={noise_budget}.JPEG"
                img_psf_img_path = os.path.join(Smpf_dir, img_psf_img_name)
                img_psf_img_pil = Image.fromarray(img_psf_img.astype(np.uint8))
                img_psf_img_pil.save(img_psf_img_path)

        # 清理不再需要的变量和内存
        del attack, at_images, org_imgs, org_labels
        torch.cuda.empty_cache()




