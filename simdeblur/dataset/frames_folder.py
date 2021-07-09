""" ************************************************
* fileName: frames_folder.py
* desc: A class of frames folder
* author: mingdeng_cao
* date: 2021/07/09 16:18
* last revised: None
************************************************ """


import os
import torch
import numpy as np
import cv2
import logging


class FramesFolder(torch.utils.data.Dataset):
    """
    Args:
        cfg(Easydict): The config file for dataset. The following attributes should be contained:
            root_input: the root path of the input videos
            num_frames: the number of input frames of the model
            overlapping: the input frames are whether overlapped
            sampling: the sampling mode.
    """

    def __init__(self, cfg):
        self.cfg = cfg

        self.frames = os.listdir(cfg.root_input)
        self.frames.sort()
        if cfg.overlapping:
            self.frames = self.frames[:-cfg.num_frames]
        else:
            self.frames = self.frames[::cfg.num_frames]

        assert self.frames, "Their is no frames in '{}'. ".format(
            self.cfg.root_input)

        logging.info(
            f"Total samples {len(self.frames)} are loaded for testing!")

    def get_frames_name(self, frame_name, num_frames, sampling="n_c"):
        """
        Args:
            frame_name: the frame's name corresponding to the idx.
            num_frames: the number of input frames of the model.
            interval: the interval when sampling.
            sampling: the sampling mode.
        """
        frame_idx, suffix = frame_name.split(".")
        frame_idx_length = len(frame_idx)
        frame_idx = int(frame_idx)

        frame_name_format = "{:0" + str(frame_idx_length) + "d}.{}"
        # when to read the frames, should pay attention to the name of frames
        input_frames_name = [frame_name_format.format(
            i, suffix) for i in range(frame_idx, frame_idx + num_frames)]

        if sampling == "n_c":
            gt_frames_name = [input_frames_name[num_frames//2]]
        elif sampling == "n_l":
            gt_frames_name = [input_frames_name[0]]
        elif sampling == "n_r":
            gt_frames_name = [input_frames_name[-1]]
        elif sampling == "n_n":
            gt_frames_name = input_frames_name

        return input_frames_name, gt_frames_name

    def read_img_opencv(self, path):
        """
        read image by opencv
        return: Numpy float32, HWC, BGR, [0,1]
        """
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print("the path is None! {} !".format(path))
        img = img.astype(np.float32) / 255.
        if img.ndim == 2:
            img = np.expand_dims(img, axis=2)
        # some images have 4 channels
        if img.shape[2] > 3:
            img = img[:, :, :3]
        return img

    def __getitem__(self, idx):
        frame_name = self.frames[idx]

        input_frames_name, gt_frames_name = self.get_frames_name(
            frame_name, self.cfg.num_frames, self.cfg.sampling)

        assert len(input_frames_name) == self.cfg.num_frames, "Wrong frames length not equal the sampling frames {}".format(
            self.cfg.num_frames)

        # Read images by opencv with format HWC, BGR, [0,1], TODO add other loading methods.
        input_frames = [self.read_img_opencv(os.path.join(
            self.cfg.root_input, frame_name)) for frame_name in input_frames_name]

        # stack and transpose with RGB style (n, c, h, w)
        input_frames = np.stack(
            input_frames, axis=0)[..., ::-1].transpose([0, 3, 1, 2])

        # To tensor with contingious array.
        input_frames = torch.from_numpy(
            np.ascontiguousarray(input_frames)).float()

        return {
            "input_frames": input_frames,
            "gt_names": gt_frames_name,
        }

    def __len__(self):
        return len(self.frames)
