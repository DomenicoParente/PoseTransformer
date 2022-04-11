import os
import os.path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms


# DataLoader for RGB-D Scenes Dataset v.2 by University of Washington

class RGBDDataset(Dataset):
    def __init__(self, img_dir, label_dir, n_segment, frame_template, label_template, n_video, frames_per_segment=1,
                 transform=None):

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.frame_template = frame_template
        self.label_template = label_template
        self.n_video = n_video
        self.n_segment = n_segment
        self.frames_per_segment = frames_per_segment

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def load_frame(self, idx, frm_idx):
        return Image.open(
            os.path.join(self.img_dir, "scene_{:02d}".format(idx + 1), self.frame_template.format(frm_idx)))

    def load_labels(self, idx):
        return open(os.path.join(self.label_dir, self.label_template.format(idx + 1)), 'r')

    def getLabel(self, idx, num_frame):
        file = open(os.path.join(self.label_dir, self.label_template.format(idx + 1)), 'r')
        label = []
        for pos, line in enumerate(file):
            if pos == num_frame:
                label = list(map(float, line.split()))
                label = np.array(label)
                # print("Scene: {:2d}, Frame: {:05d}, Pose: ".format(idx+1, num_frame), label)
        if len(label) != 7:
            print("ERROR - The label contains {:d} values".format(len(label)))
        return label

    def getN_frames(self, idx):
        fp = self.load_labels(idx)
        count = 0
        for count, l in enumerate(fp):
            pass
        return count + 1

    def getStartIdx(self, idx):
        max_valid_start_index = (self.getN_frames(idx) - self.frames_per_segment) // self.n_segment
        start_indices = np.multiply(list(range(self.n_segment)), max_valid_start_index) + \
                        np.random.randint(max_valid_start_index, size=self.n_segment)
        return start_indices

    def __len__(self):
        return self.n_video

    def __getitem__(self, idx):
        frames = list()
        labels = list()
        l_frames = []

        if self.getN_frames(idx) < self.n_segment * self.frames_per_segment:
            print("ERROR. The number of frames in dataset is smaller than the number required.")

        for start_idx in self.getStartIdx(idx):
            frame_idx = start_idx

            for i in range(self.frames_per_segment):
                f = self.load_frame(idx, frame_idx)
                frames.append(f)
                # For each segment it takes the last label
                if i == (self.frames_per_segment - 1):
                    label = self.getLabel(idx, frame_idx)
                    labels.append(label)

                if frame_idx < self.getN_frames(idx) - 1:
                    frame_idx += 1

        if self.transform is not None:
            for fr in frames:
                fr = self.transform(fr)
                l_frames.append(fr)

        # Final output should be a tensor of shape [C, T, H, W]
        l_frames = torch.stack(l_frames)
        l_frames = torch.permute(l_frames, [1, 0, 2, 3])
        labels = np.array(labels)
        # print(l_frames.size())
        return l_frames, labels


class RGBDDatasetv2(Dataset):
    def __init__(self, img_dir, label_dir, n_segment, frame_template, label_template, n_video, frames_per_segment=1,
                 transform=None):

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.frame_template = frame_template
        self.label_template = label_template
        self.n_video = n_video
        self.n_segment = n_segment
        self.frames_per_segment = frames_per_segment

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def load_frame(self, idx, frm_idx):
        return Image.open(
            os.path.join(self.img_dir, "scene_{:02d}".format(idx + 1), self.frame_template.format(frm_idx)))

    def load_labels(self, idx):
        return open(os.path.join(self.label_dir, self.label_template.format(idx + 1)), 'r')

    def getLabel(self, idx, num_frame):
        file = open(os.path.join(self.label_dir, self.label_template.format(idx + 1)), 'r')
        label = []
        for pos, line in enumerate(file):
            if pos == num_frame:
                label = list(map(float, line.split()))
                label = np.array(label)
                # print("Scene: {:2d}, Frame: {:05d}, Pose: ".format(idx+1, num_frame), label)
        if len(label) != 7:
            print("ERROR - The label contains {:d} values".format(len(label)))
        return label

    def getN_frames(self, idx):
        fp = self.load_labels(idx)
        count = 0
        for count, l in enumerate(fp):
            pass
        return count + 1

    def __len__(self):
        return self.n_video

    def __getitem__(self, idx):
        frames = list()
        labels = list()
        l_frames = []
        n_frames = self.frames_per_segment * self.n_segment

        if self.getN_frames(idx) < n_frames:
            print("ERROR. The number of frames in dataset is smaller than the number required.")

        max_valid_start_index = self.getN_frames(idx) - n_frames
        frame_idx = np.random.randint(0, max_valid_start_index)
        for i in range(n_frames):
            f = self.load_frame(idx, (frame_idx + i))
            frames.append(f)
            # For each segment it takes the last label
            if i % 2 == 0:
                label = self.getLabel(idx, (frame_idx + i))
                labels.append(label)

        if self.transform is not None:
            for fr in frames:
                fr = self.transform(fr)
                l_frames.append(fr)

        # Final output should be a tensor of shape [C, T, H, W]
        l_frames = torch.stack(l_frames)
        l_frames = torch.permute(l_frames, [1, 0, 2, 3])
        labels = np.array(labels)
        # print(l_frames.size())
        return l_frames, labels
