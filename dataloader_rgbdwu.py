import os
import os.path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import dataAugmentation
import utils


# DataLoader for RGB-D Scenes Dataset v.2 by University of Washington
class RGBDDataset(Dataset):
    """
    It divides the frames in groups and takes a fixed number of frames from each group, this means that the frames
    can be very spaced apart from each other. Reference: https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch
    """

    def __init__(self, img_dir, label_dir, n_segment, frame_template, label_template, n_video, frames_per_segment=1,
                 data_augmentation=False, transform=None):

        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.frame_template = frame_template
        self.label_template = label_template
        self.n_video = n_video
        self.n_segment = n_segment
        self.frames_per_segment = frames_per_segment
        self.data_augmentation = data_augmentation

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
                # constrain to hemisphere
                for i in range(4):
                    label[i] = np.sign(label[0]) * label[i]
                # correct pose format: position + orientation (quaternion with scalar value at the end)
                label = np.array([label[4], label[5], label[6], label[1], label[2], label[3], label[0]])

                # print("Scene: {:2d}, Frame: {:05d}, Pose: ".format(idx+1, num_frame), label)
        if len(label) != 7:
            raise Exception("ERROR - The label contains {:d} values".format(len(label)))
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
        temp = []

        if self.getN_frames(idx) < self.n_segment * self.frames_per_segment:
            raise Exception("ERROR. The number of frames in dataset is smaller than the number required.")

        for start_idx in self.getStartIdx(idx):
            frame_idx = start_idx

            for i in range(self.frames_per_segment):
                f = self.load_frame(idx, frame_idx)
                frames.append(f)
                # For each segment it takes the initial and last label
                if i == 0:
                    label1 = self.getLabel(idx, frame_idx)
                if i == (self.frames_per_segment - 1):
                    label2 = self.getLabel(idx, frame_idx)

                if frame_idx < self.getN_frames(idx) - 1:
                    frame_idx += 1

            label = utils.glob_to_rel(label1, label2)
            labels.append(label)

        if self.data_augmentation:
            aug = dataAugmentation.ImageAugmentation()
            for fr in frames:
                fr = aug(fr)
                temp.append(fr)

        if self.transform is not None:
            for fr in frames:
                fr = self.transform(fr)
                l_frames.append(fr)

        # Final output is a tensor of shape [T, C, H, W]
        l_frames = torch.stack(l_frames)
        l_frames = torch.permute(l_frames, [1, 0, 2, 3])
        labels = np.array(labels)
        # print(l_frames.size())
        return l_frames, labels


class RGBDDataset_v2(RGBDDataset):
    """This dataloader takes n consecutive frames (overlapping last pose of previous item with the initial pose of the next item) each time until the frames in the current scene are finished.
     Then it passes to the next scene. """

    def __init__(self, img_dir, label_dir, n_segment, frame_template, label_template, n_video, frames_per_segment=1,
                 data_augmentation=False, transform=None):

        super().__init__(img_dir, label_dir, n_segment, frame_template, label_template, n_video, frames_per_segment,
                         data_augmentation, transform)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.frame_template = frame_template
        self.label_template = label_template
        self.n_video = n_video
        self.frames_per_segment = frames_per_segment
        self.n_frames = n_segment * frames_per_segment
        self.data_augmentation = data_augmentation

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        count = 0
        for i in range(self.n_video):
            count += self.getN_frames(i) // self.n_frames
        return count

    def __getitem__(self, idx):
        frames = list()
        labels = list()
        l_frames = []
        temp = []
        video_index = self.n_video + 1
        count_p = 0
        count = 0

        for j in range(self.n_video):
            count_p = count
            count += (self.getN_frames(j) // self.n_frames)
            if idx < count:
                video_index = j
                break

        if video_index > self.n_video:
            raise Exception("Error Data-loader index.")

        if self.getN_frames(video_index) < self.n_frames:
            raise Exception("ERROR. The number of frames in dataset is smaller than the number required.")

        frame_idx = (idx - count_p) * self.n_frames
        for i in range(self.n_frames):
            f = self.load_frame(video_index, (frame_idx + i))
            frames.append(f)
            # For each segment it takes the initial and last labels and calculate relative pose
            if (i + 1) % self.frames_per_segment == 0:
                label1 = self.getLabel(video_index, (frame_idx + i + 1 - self.frames_per_segment))
                label2 = self.getLabel(video_index, (frame_idx + i))
                label = utils.glob_to_rel(label1, label2)
                labels.append(label)

        if self.data_augmentation:
            aug = dataAugmentation.ImageAugmentation()
            for fr in frames:
                fr = aug(fr)
                temp.append(fr)
        else:
            temp = frames

        if self.transform is not None:
            for fr in temp:
                fr = self.transform(fr)
                l_frames.append(fr)

        l_frames = torch.stack(l_frames)
        l_frames = torch.permute(l_frames, [1, 0, 2, 3])
        labels = np.array(labels)
        return l_frames, labels


class RGBDDataset_overlapping(RGBDDataset):
    """This dataloader takes n consecutive frames (n/t poses) each time sliding of 1 frame
     until the frames in the current scene are finished. Then it passes to the next scene. """

    def __init__(self, img_dir, label_dir, n_segment, frame_template, label_template, n_video, frames_per_segment=1,
                 data_augmentation=False, transform=None):

        super().__init__(img_dir, label_dir, n_segment, frame_template, label_template, n_video, frames_per_segment,
                         data_augmentation, transform)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.frame_template = frame_template
        self.label_template = label_template
        self.n_video = n_video
        self.frames_per_segment = frames_per_segment
        self.n_frames = n_segment * frames_per_segment
        self.data_augmentation = data_augmentation

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        count = 0
        for i in range(self.n_video):
            count += self.getN_frames(i) // self.n_segment
        return count

    def __getitem__(self, idx):
        frames = list()
        labels = list()
        l_frames = []
        temp = []
        video_index = self.n_video + 1
        count_p = 0
        count = 0

        for j in range(self.n_video):
            count_p = count
            count += self.getN_frames(j) // self.n_segment
            if idx < count:
                video_index = j
                break

        if video_index > self.n_video:
            raise Exception("Error Data-loader index.")

        if self.getN_frames(video_index) < self.n_frames:
            raise Exception("ERROR. The number of frames in dataset is smaller than the number required.")

        frame_idx = (idx - count_p)
        frame_idx *= self.n_segment
        for _ in range(self.n_segment):
            for j in reversed(range(self.frames_per_segment)):
                if frame_idx - j < 0:
                    f = self.load_frame(video_index, 0)
                else:
                    f = self.load_frame(video_index, (frame_idx - j))
                if j == 0:
                    if frame_idx -self.frames_per_segment < 0:
                        label1 = self.getLabel(video_index, 0)
                    else:
                        label1 = self.getLabel(video_index, (frame_idx + 1 - self.frames_per_segment))
                    label2 = self.getLabel(video_index, (frame_idx - j))
                    label = utils.glob_to_rel(label1, label2)
                    labels.append(label)
                frames.append(f)
            frame_idx += 1

        if self.data_augmentation:
            aug = dataAugmentation.ImageAugmentation()
            for fr in frames:
                fr = aug(fr)
                temp.append(fr)
        else:
            temp = frames

        if self.transform is not None:
            for fr in temp:
                fr = self.transform(fr)
                l_frames.append(fr)

        l_frames = torch.stack(l_frames)
        l_frames = torch.permute(l_frames, [1, 0, 2, 3])
        labels = np.array(labels)
        return l_frames, labels


class RGBDDataset_test(RGBDDataset):
    """Dato loader for testing: it takes all frames from a chosen scene (n frames per time) """

    def __init__(self, img_dir, label_dir, n_segment, frame_template, label_template, n_video, frames_per_segment=1,
                 scene=14, transform=None):

        super().__init__(img_dir, label_dir, n_segment, frame_template, label_template, n_video, frames_per_segment,
                         transform)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.frame_template = frame_template
        self.label_template = label_template
        self.n_video = n_video
        self.index = scene - 1
        self.n_frames = n_segment * frames_per_segment
        self.frames_per_segment = frames_per_segment

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return self.getN_frames(self.index) // self.n_frames

    def __getitem__(self, idx):
        frames = list()
        labels = list()
        l_frames = []

        if self.getN_frames(self.index) < self.n_frames:
            raise Exception("ERROR. The number of frames in dataset is smaller than the number required.")

        frame_idx = idx * self.n_frames
        for i in range(self.n_frames):
            f = self.load_frame(self.index, (frame_idx + i))
            frames.append(f)
            # For each segment it takes the last label
            if (i + 1) % self.frames_per_segment == 0:
                label1 = self.getLabel(self.index, (frame_idx + i + 1 - self.frames_per_segment))
                label2 = self.getLabel(self.index, (frame_idx + i))
                label = utils.glob_to_rel(label1, label2)
                labels.append(label)

        if self.transform is not None:
            for fr in frames:
                fr = self.transform(fr)
                l_frames.append(fr)

        l_frames = torch.stack(l_frames)
        l_frames = torch.permute(l_frames, [1, 0, 2, 3])
        labels = np.array(labels)
        # print(l_frames.size())
        return l_frames, labels


class RGBDDataset_test_overlapping(RGBDDataset):
    """Dato loader for testing: it takes all frames from a chosen scene (n frames per time) """

    def __init__(self, img_dir, label_dir, n_segment, frame_template, label_template, n_video, frames_per_segment=1,
                 scene=14, transform=None):

        super().__init__(img_dir, label_dir, n_segment, frame_template, label_template, n_video, frames_per_segment,
                         transform)
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.transform = transform
        self.frame_template = frame_template
        self.label_template = label_template
        self.n_video = n_video
        self.index = scene - 1
        self.n_frames = n_segment * frames_per_segment
        self.frames_per_segment = frames_per_segment

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return self.getN_frames(self.index) // self.n_segment

    def __getitem__(self, idx):
        frames = list()
        labels = list()
        l_frames = []

        if self.getN_frames(self.index) < self.n_frames:
            print("ERROR. The number of frames in dataset is smaller than the number required.")

        frame_idx = idx
        frame_idx *= self.n_segment
        for _ in range(self.n_segment):
            for j in reversed(range(self.frames_per_segment)):
                if frame_idx - j < 0:
                    f = self.load_frame(self.index, 0)
                else:
                    f = self.load_frame(self.index, (frame_idx - j))
                if j == 0:
                    if frame_idx - self.frames_per_segment < 0:
                        label1 = self.getLabel(self.index, 0)
                    else:
                        label1 = self.getLabel(self.index, (frame_idx + 1 - self.frames_per_segment))
                    label2 = self.getLabel(self.index, (frame_idx - j))
                    label = utils.glob_to_rel(label1, label2)
                    labels.append(label)
                frames.append(f)
            frame_idx += 1

        if self.transform is not None:
            for fr in frames:
                fr = self.transform(fr)
                l_frames.append(fr)

        l_frames = torch.stack(l_frames)
        l_frames = torch.permute(l_frames, [1, 0, 2, 3])
        labels = np.array(labels)
        # print(l_frames.size())
        return l_frames, labels


