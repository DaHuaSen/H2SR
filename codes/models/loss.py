import numpy as np
import cv2
import torch
import torch.nn.functional as F


def dft_tensor(img, size):
    gray_img = np.dot(img.detach().cpu().numpy().transpose(1, 2, 0), [0.2989, 0.5870, 0.1140])
    gray_img = np.float32(gray_img)
    dft = cv2.dft(gray_img, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    row, col = gray_img.shape
    crow, ccol = row // 2, col // 2

    hmask = np.ones((row, col, 2), np.uint8)
    hmask[crow - size:crow + size, ccol - size: ccol + size] = 0

    fshift = dft_shift * hmask
    f_ishift = np.fft.ifftshift(fshift)
    idft_img = cv2.idft(f_ishift)
    idft_img = cv2.magnitude(idft_img[:, :, 0], idft_img[:, :, 1])
    tensor = torch.tensor(idft_img, dtype=torch.float32)
    return tensor


def cal_dft(fake, var, size=10):
    loss = 0
    for i in range(len(fake)):
        loss_temp = F.l1_loss(dft_tensor(fake[i], size), dft_tensor(var[i], size))
        loss += loss_temp
    return loss


def dct_tensor(img, size):
    gray_img = np.dot(img.detach().cpu().numpy().transpose(1, 2, 0), [0.2989, 0.5870, 0.1140])
    dct_img = cv2.dct(np.float32(gray_img))

    dct_img[:size, :size] = 1
    # x, y = np.meshgrid(np.arange(0, dct_img.shape[0]), np.arange(0, dct_img.shape[1]))
    # distance = np.sqrt(x ** 2 + y ** 2)
    # dct_img[(distance <= size) & (x <= size) & (y <= size)] = 1

    idct_img = cv2.idct(dct_img)
    tensor = torch.tensor(idct_img, dtype=torch.float32)
    return tensor


def cal_dct(fake, var, size=10):
    loss = 0
    for i in range(len(fake)):
        loss_temp = F.l1_loss(dct_tensor(fake[i], size), dct_tensor(var[i], size))
        loss += loss_temp
    return loss
