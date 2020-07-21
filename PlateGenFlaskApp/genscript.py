import os
import glob
import argparse

import cv2
import numpy as np
import string

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

from PIL import Image, ImageDraw, ImageFont
import imageio

device = ""
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Generator, self).__init__()

        # reduce to features
        self.c0 = nn.Conv2d(in_channels, 64, 4, stride=2, padding=1)
        self.c1 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.c2 = nn.Conv2d(128, 256, 4, stride=2, padding=1)
        self.c3 = nn.Conv2d(256, 512, 4, stride=2, padding=1)

        # upsample to image

        self.d3 = nn.ConvTranspose2d(512, 256, 4, stride=2, padding=1)
        self.d2 = nn.ConvTranspose2d(512, 128, 4, stride=2, padding=1)
        self.d1 = nn.ConvTranspose2d(256, 64, 4, stride=2, padding=1)
        self.d0 = nn.ConvTranspose2d(128, out_channels, 4, stride=2, padding=1)

        self.bnc1 = nn.BatchNorm2d(128)
        self.bnc2 = nn.BatchNorm2d(256)
        self.bnc3 = nn.BatchNorm2d(512)

        self.bnd3 = nn.BatchNorm2d(256)
        self.bnd2 = nn.BatchNorm2d(128)
        self.bnd1 = nn.BatchNorm2d(64)

    def forward(self, x):
        en0 = self.c0(x)
        en1 = self.bnc1(self.c1(F.leaky_relu(en0, negative_slope=0.2)))
        en2 = self.bnc2(self.c2(F.leaky_relu(en1, negative_slope=0.2)))
        en3 = self.bnc3(self.c3(F.leaky_relu(en2, negative_slope=0.2)))

        de3 = self.bnd3(self.d3(F.relu(en3)))
        de2 = F.dropout(self.bnd2(self.d2(F.relu(torch.cat((en2, de3), 1)))))
        de1 = self.bnd1(self.d1(F.relu(torch.cat((en1, de2), 1))))

        de0 = torch.tanh(self.d0(F.relu(torch.cat((en0, de1), 1))))

        return de0

def make_plate_images(num: int, istate = None,
                      fonts_folder_path: string = './ttf_fonts/'):

    G = Generator(1, 1).to(device)
    G.load_state_dict(torch.load('./trained_shader_state_dictionary.pth'))
    G.eval()
    try:
        os.mkdir('./generated_images')
    except Exception as e:
        print(e)

    # clearing image folder
    files = glob.glob('./generated_images/*')
    for file in files:
        os.remove(file)

    # https://en.wikipedia.org/wiki/List_of_Regional_Transport_Office_districts_in_India
    # incorporated upper caps, ignored complications with some states
    STATE_CODES = [['AP', 39], ['AR', 22], ['AS', 30], ['BR', 57], ['CG', 30], ['GA', 12], ['GJ', 38], ['HR', 99],
                   ['HP', 97], ['JH', 24], ['KA', 70], ['KL', 86], ['MP', 74],
                   ['MH', 50], ['MN', 7], ['ML', 10], ['MZ', 8], ['NL', 8], ['OD', 35], ['PB', 91], ['RJ', 55],
                   ['SK', 4], ['TN', 99], ['TS', 36], ['TR', 8], ['UP', 96],
                   ['UK', 20], ['WB', 99], ['AN', 1], ['CH', 4], ['DD', 3], ['DL', 16], ['JK', 22], ['LA', 2],
                   ['LD', 9], ['PY', 5]]
    global state

    found_match = False
    for i in range(len(STATE_CODES)):
        if STATE_CODES[i][0] == istate:
            state = [istate, STATE_CODES[i][1]]
            found_match = True


    for i in tqdm(range(num)):
        # AA 11 BB 2222
        letters = np.random.choice(list(string.ascii_uppercase)) + np.random.choice(
            list(string.ascii_uppercase))  # random BB

        if not (found_match):  # random valid AA
            state = STATE_CODES[np.random.randint(len(STATE_CODES))]

        lp_str = state[
                     0] + " " + f"{np.random.randint(0, state[1] + 1):02}" + " " + letters + " " + f"{np.random.randint(10000):04} "

        image = Image.new('RGB', (128, 32), color=(0,) * 3)
        draw = ImageDraw.Draw(image)

        font_size = 2
        font_path = np.random.choice(glob.glob(fonts_folder_path + '*.ttf'))
        font = ImageFont.truetype(font_path, font_size)
        while (font.getsize(lp_str)[0] / 2 < 57 and font.getsize(lp_str)[1] / 2 < 8):
            font_size += 2
            font = ImageFont.truetype(font_path, font_size)

        draw.text((64 - font.getsize(lp_str)[0] / 2, 16 - font.getsize(lp_str)[1] / 2), lp_str, color=(255,) * 3,
                  font=font)
        image.save('image.png')
        image = cv2.imread('image.png', cv2.IMREAD_GRAYSCALE)

        y = image
        y = y.astype(np.float32)
        y = (y - 127.5) / 127.5
        y = torch.tensor(y).to(device)
        y = y.unsqueeze(0).unsqueeze(0)

        out = G(y)

        out = out.squeeze(0).squeeze(0)
        out = cv2.blur((((out.detach().cpu().numpy() + 1) / 2) * 255).astype(np.uint8), (3, 3))
        y = out
        random_jitter_matrix = np.array(([1.0 - np.random.uniform(0, 0.1), 0, 0],
                                         [0, 1.0 - np.random.uniform(0, 0.1), 0],
                                         [0, 0, 1.0 - np.random.uniform(0, 0.1)]))
        y = cv2.warpPerspective(y, random_jitter_matrix, dsize=(128, 32), borderValue=np.ones(3) * 0)
        y = cv2.warpAffine(y, cv2.getRotationMatrix2D((16, 64), np.random.uniform(-2, 2), 1), dsize=(128, 32))

        imageio.imsave('./generated_images/' + lp_str + f"__{i}" + '.png', y)


    os.remove('image.png')
    endmessage = ""
    if not found_match:
        endmessage += "State code entry blank or invalid. Included all. <br>"

    endmessage += "Download prepared."
    return endmessage