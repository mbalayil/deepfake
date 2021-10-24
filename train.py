#!usr/bin/env python3

import cv2
import time
import math
import os
import dlib
import numpy as np

import torch
import torch.nn as nn
from torch.nn.functional import pad
from torch.nn.parameter import Parameter
from torch.nn.modules import Module
from torch.nn.modules.utils import _single, _pair, _triple
import torch.nn.functional as F
import torch.optim as optim

from umeyama import umeyama

def frames_from_videos(input_path, save_path):
    cap = cv2.VideoCapture(input_path)
    try:
        os.mkdir(save_path)
    except OSError:
        pass
    n = 0
    fourcc = cv2.VideoWriter_fourcc("M", "J", "P", "G")
    frame_width = int(cap.get(3))
    # frame_width = int(cap.get(50))
    frame_height = int(cap.get(4))
    # frame_height = int(cap.get(50))
    out = cv2.VideoWriter("liu_out.avi", fourcc, 10, (frame_width, frame_height))
    while (cap.isOpened()) and n < 500:
        ret, frame = cap.read()
        if not ret:
            break
        save_images = os.path.join(save_path, str(n) + ".jpg")
        cv2.imwrite(save_images, frame)
        if ret == True:
            cv2.imshow("frame", frame)
            # time.sleep(1)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            # break
        n = n + 1
        print(n)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def faces_from_frames(frames_path, face_path):

    # frames_path = os.path.join(os.path.realpath('.'), frames_folder)
    try:
        os.mkdir(face_path)
    except OSError:
        pass
    print(frames_path)
    print(face_path)
    pictures = os.listdir(frames_path)

    detector = dlib.get_frontal_face_detector()

    # print(pictures)
    for f in pictures:
        img = cv2.imread(os.path.join(frames_path, f), cv2.IMREAD_COLOR)
        b, g, r = cv2.split(img)
        img2 = cv2.merge([r, g, b])
        # img = rotate(img2)
        # cv2.imwrite(face_path+f[:-4]+"_face.jpg", img)
        dets = detector(img2, 1)
        print("Number of faces detected: {}".format(len(dets)))

        for idx, face in enumerate(dets):
            left = face.left()
            top = face.top()
            right = face.right()
            bot = face.bottom()
            crop_img = img[top:bot, left:right]
            save_path = os.path.join(face_path, f[:-4] + ".jpg")
            if crop_img.size != 0:
                cv2.imwrite(save_path, crop_img)


def get_image_paths(directory):
    # Get all the image paths in the given directory
    return [
        x.path
        for x in os.scandir(directory)
        if x.name.endswith(".jpg") or x.name.endswith(".png") or x.name.endswith(".JPG")
    ]


def load_images(paths):
    # Load images from each image paths from 'paths' and resize it
    images_resized = (cv2.resize(cv2.imread(p), (256, 256)) for p in paths)
    for i, im in enumerate(images_resized):
        if i == 0:
            images = np.empty((len(paths),) + im.shape, dtype=im.dtype)
        images[i] = im
    return images


class _ConvNd(Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
    ):
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if transposed:
            self.weight = Parameter(
                torch.Tensor(in_channels, out_channels // groups, *kernel_size)
            )
        else:
            self.weight = Parameter(
                torch.Tensor(out_channels, in_channels // groups, *kernel_size)
            )
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def __repr__(self):
        s = (
            "{name}({in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        s += ")"
        return s.format(name=self.__class__.__name__, **self.__dict__)


def conv2d_same_padding(
    input, weight, bias=None, stride=1, padding=1, dilation=1, groups=1
):

    input_rows = input.size(2)
    filter_rows = weight.size(2)
    effective_filter_size_rows = (filter_rows - 1) * dilation[0] + 1
    out_rows = (input_rows + stride[0] - 1) // stride[0]
    padding_needed = max(
        0, (out_rows - 1) * stride[0] + effective_filter_size_rows - input_rows
    )
    padding_rows = max(
        0, (out_rows - 1) * stride[0] + (filter_rows - 1) * dilation[0] + 1 - input_rows
    )
    rows_odd = padding_rows % 2 != 0
    padding_cols = max(
        0, (out_rows - 1) * stride[0] + (filter_rows - 1) * dilation[0] + 1 - input_rows
    )
    cols_odd = padding_rows % 2 != 0

    if rows_odd or cols_odd:
        input = pad(input, [0, int(cols_odd), 0, int(rows_odd)])

    return F.conv2d(
        input,
        weight,
        bias,
        stride,
        padding=(padding_rows // 2, padding_cols // 2),
        dilation=dilation,
        groups=groups,
    )


class Conv2d(_ConvNd):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
    ):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
        )

    def forward(self, input):
        return conv2d_same_padding(
            input,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class _ConvLayer(nn.Sequential):
    def __init__(self, input_features, output_features):
        super(_ConvLayer, self).__init__()
        self.add_module(
            "conv2", Conv2d(input_features, output_features, kernel_size=5, stride=2)
        )
        self.add_module("leakyrelu", nn.LeakyReLU(0.1, inplace=True))


class _UpScale(nn.Sequential):
    def __init__(self, input_features, output_features):
        super(_UpScale, self).__init__()
        self.add_module(
            "conv2_", Conv2d(input_features, output_features * 4, kernel_size=3)
        )
        self.add_module("leakyrelu", nn.LeakyReLU(0.1, inplace=True))
        self.add_module("pixelshuffler", _PixelShuffler())


class Flatten(nn.Module):
    def forward(self, input):
        output = input.view(input.size(0), -1)
        return output


class Reshape(nn.Module):
    def forward(self, input):
        output = input.view(-1, 1024, 4, 4)  # channel * 4 * 4

        return output


class _PixelShuffler(nn.Module):
    def forward(self, input):
        batch_size, c, h, w = input.size()
        rh, rw = (2, 2)
        oh, ow = h * rh, w * rw
        oc = c // (rh * rw)
        out = input.view(batch_size, rh, rw, oc, h, w)
        out = out.permute(0, 3, 4, 1, 5, 2).contiguous()
        out = out.view(batch_size, oc, oh, ow)  # channel first

        return out


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()

        self.encoder = nn.Sequential(
            _ConvLayer(3, 128),
            _ConvLayer(128, 256),
            _ConvLayer(256, 512),
            _ConvLayer(512, 1024),
            Flatten(),
            nn.Linear(1024 * 4 * 4, 1024),
            nn.Linear(1024, 1024 * 4 * 4),
            Reshape(),
            _UpScale(1024, 512),
        )

        self.decoder_A = nn.Sequential(
            _UpScale(512, 256),
            _UpScale(256, 128),
            _UpScale(128, 64),
            Conv2d(64, 3, kernel_size=5, padding=1),
            nn.Sigmoid(),
        )

        self.decoder_B = nn.Sequential(
            _UpScale(512, 256),
            _UpScale(256, 128),
            _UpScale(128, 64),
            Conv2d(64, 3, kernel_size=5, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x, select="A"):
        if select == "A":
            out = self.encoder(x)
            out = self.decoder_A(out)
        else:
            out = self.encoder(x)
            out = self.decoder_B(out)
        return out


def get_device():
    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    return device


random_transform_args = {
    "rotation_range": 10,
    "zoom_range": 0.05,
    "shift_range": 0.05,
    "random_flip": 0.4,
}


def random_transform(image, rotation_range, zoom_range, shift_range, random_flip):
    h, w = image.shape[0:2]
    rotation = np.random.uniform(-rotation_range, rotation_range)
    scale = np.random.uniform(1 - zoom_range, 1 + zoom_range)
    tx = np.random.uniform(-shift_range, shift_range) * w
    ty = np.random.uniform(-shift_range, shift_range) * h
    mat = cv2.getRotationMatrix2D((w // 2, h // 2), rotation, scale)
    mat[:, 2] += (tx, ty)
    result = cv2.warpAffine(image, mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    if np.random.random() < random_flip:
        result = result[:, ::-1]
    return result


# get pair of random warped images from aligened face image
def random_warp(image):
    assert image.shape == (256, 256, 3)
    range_ = np.linspace(128 - 80, 128 + 80, 5)
    mapx = np.broadcast_to(range_, (5, 5))
    mapy = mapx.T

    mapx = mapx + np.random.normal(size=(5, 5), scale=5)
    mapy = mapy + np.random.normal(size=(5, 5), scale=5)

    interp_mapx = cv2.resize(mapx, (80, 80))[8:72, 8:72].astype("float32")
    interp_mapy = cv2.resize(mapy, (80, 80))[8:72, 8:72].astype("float32")

    warped_image = cv2.remap(image, interp_mapx, interp_mapy, cv2.INTER_LINEAR)

    src_points = np.stack([mapx.ravel(), mapy.ravel()], axis=-1)
    dst_points = np.mgrid[0:65:16, 0:65:16].T.reshape(-1, 2)
    mat = umeyama(src_points, dst_points, True)[0:2]

    target_image = cv2.warpAffine(image, mat, (64, 64))

    return warped_image, target_image


def get_training_data(images, batch_size):
    indices = np.random.randint(len(images), size=batch_size)
    # train_data = []
    for i, index in enumerate(indices):
        image = images[index]
        image = random_transform(image, **random_transform_args)
        warped_img, target_img = random_warp(image)

        if i == 0:
            warped_images = np.empty((batch_size,) + warped_img.shape, warped_img.dtype)
            target_images = np.empty((batch_size,) + target_img.shape, warped_img.dtype)
            # train_data = np.empty((batch_size,) + image.shape, image.dtype)
        # train_data[i] = image
        warped_images[i] = warped_img
        target_images[i] = target_img
    return warped_images, target_images
    # return train_data


def var_to_np(img_var):
    return img_var.data.cpu().numpy()


if __name__ == "__main__":
    # Read input videos and convert to frames
    Input_Path1 = "inputs/obama_short.mp4"
    input_path1 = os.path.join(os.path.realpath("."), Input_Path1)
    save_path1 = os.path.join(os.path.dirname(input_path1), "obama_frames/")
    frames_from_videos(input_path1, save_path1)

    Input_Path2 = "inputs/trump_short.mp4"
    input_path2 = os.path.join(os.path.realpath("."), Input_Path2)
    save_path2 = os.path.join(os.path.dirname(input_path2), "trump_frames/")
    frames_from_videos(input_path2, save_path2)

    # Identify faces from frames
    frames_folder1 = "inputs/obama_frames"
    input_path1 = os.path.join(os.path.realpath("."), frames_folder1)
    face_folder1 = "inputs/obama_faces"
    save_path1 = os.path.join(os.path.realpath("."), face_folder1)
    faces_from_frames(input_path1, save_path1)

    frames_folder2 = "inputs/trump_frames"
    input_path2 = os.path.join(os.path.realpath("."), frames_folder2)
    face_folder2 = "inputs/trump_faces"
    save_path2 = os.path.join(os.path.realpath("."), face_folder2)
    faces_from_frames(input_path2, save_path2)

    # Train the autoencoder
    # Create model
    model = Autoencoder()

    # Set hyperparameters
    log_interval = 100
    batch_size = 64
    epochs = 10000
    device = get_device()
    model.to(device)
    if device == "cuda:0":
        print("==Using GPU to train==")
        device = torch.device("cuda:0")
        # cudnn.benchmark = True
    else:
        print("==Using CPU to train==")

    print("==Try resume from checkpoint==")
    if os.path.isdir("checkpoint"):
        try:
            checkpoint = torch.load("./checkpoint/autoencoder.t7")
            model.load_state_dict(checkpoint["state"])
            start_epoch = checkpoint["epoch"]
            print("==Load last checkpoint data==")
        except FileNotFoundError:
            print("Can't found autoencoder.t7")
    else:
        start_epoch = 1
        print("==Start from scratch==")

    criterion = nn.L1Loss()
    optimizer_1 = optim.Adam(
        [
            {"params": model.encoder.parameters()},
            {"params": model.decoder_A.parameters()},
        ],
        lr=5e-5,
        betas=(0.5, 0.999),
    )
    optimizer_2 = optim.Adam(
        [
            {"params": model.encoder.parameters()},
            {"params": model.decoder_B.parameters()},
        ],
        lr=5e-5,
        betas=(0.5, 0.999),
    )

    # Get training data paths
    print("==Get training data==")
    faces_pathsA = get_image_paths(save_path1)
    faces_pathsB = get_image_paths(save_path2)
    face_imagesA = load_images(faces_pathsA) / 255
    face_imagesB = load_images(faces_pathsB) / 255

    for epoch in range(start_epoch, epochs):
        # Get 1 batch of training data for A and B
        warped_A, target_A = get_training_data(face_imagesA, batch_size)
        warped_B, target_B = get_training_data(face_imagesB, batch_size)
        # Convert to tensor
        warped_A = torch.from_numpy(warped_A.transpose((0, 3, 1, 2)))
        target_A = torch.from_numpy(target_A.transpose((0, 3, 1, 2)))
        warped_B = torch.from_numpy(warped_B.transpose((0, 3, 1, 2)))
        target_B = torch.from_numpy(target_B.transpose((0, 3, 1, 2)))
        if device == "cuda:0":
            warped_A = warped_A.to(device).float()
            target_A = target_A.to(device).float()
            warped_B = warped_B.to(device).float()
            target_B = target_B.to(device).float()
        else:
            warped_A = warped_A.float()
            target_A = target_A.float()
            warped_B = warped_B.float()
            target_B = target_B.float()

        # images_A = get_training_data(face_images1, batch_size)
        # images_B = get_training_data(face_images2, batch_size)
        optimizer_1.zero_grad()
        optimizer_2.zero_grad()

        # Train
        warped_A = model(warped_A, "A")
        warped_B = model(warped_B, "B")

        # Calculate loss
        loss1 = criterion(warped_A, target_A)
        loss2 = criterion(warped_B, target_B)
        loss = loss1.item() + loss2.item()
        loss1.backward()
        loss2.backward()
        optimizer_1.step()
        optimizer_2.step()
        print("epoch: {}, lossA:{}, lossB:{}".format(epoch, loss1.item(), loss2.item()))

        if epoch % log_interval == 0:

            test_A_ = target_A[0:14]
            test_B_ = target_B[0:14]
            test_A = var_to_np(target_A[0:14])
            test_B = var_to_np(target_B[0:14])
            # print("input size is {}".format(test_B_.size()))
            print("===> Saving models...")
            state = {"state": model.state_dict(), "epoch": epoch}
            if not os.path.isdir("checkpoint"):
                os.mkdir("checkpoint")
            torch.save(state, "./checkpoint/autoencoder.t7")
