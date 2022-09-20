import os
import os.path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torch
from torchvision import transforms
import dataAugmentation
import utils


# DataLoader for 7 Scenes Dataset by Microsoft Research
class SevenScenesDataset(Dataset):
    """It divides the frames in groups and takes a fixed number of frames from each group, this means that the frames
    can be very spaced apart from each other. Reference: https://github.com/RaivoKoot/Video-Dataset-Loading-Pytorch
    """

    def __init__(self, scenes_dir, n_segment, frame_template, label_template, n_scenes, mode,
                 frames_per_segment=1, data_augmentation=False, transform=None):

        self.scenes_dir = scenes_dir
        self.transform = transform
        self.frame_template = frame_template
        self.label_template = label_template
        self.n_scenes = n_scenes
        self.n_segment = n_segment
        self.frames_per_segment = frames_per_segment
        self.data_augmentation = data_augmentation
        self.mode = mode
        self.scenes_list = self.get_scenes()

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def get_scenes(self):
        scenes_list = []
        file = open(os.path.join(self.scenes_dir, "scenes.txt"))
        for line in file:
            scenes_list.append(line[:-1])
        return scenes_list

    def get_sequences(self, scene):
        sequences_list = []
        if self.mode == "test":
            filename = "TestSplit.txt"
        else:
            filename = "TrainSplit.txt"
        file = open(os.path.join(self.scenes_dir, scene, filename))
        for line in file:
            sequences_list.append(line[:-1])
        return sequences_list

    def load_frame(self, scene, sequence, frm_idx):
        return Image.open(
            os.path.join(self.scenes_dir, scene, sequence, self.frame_template.format(frm_idx)))

    def getLabel(self, scene, sequence, idx):
        file = open(os.path.join(self.scenes_dir, scene, sequence, self.label_template.format(idx)), 'r')
        label = np.loadtxt(file).flatten()[:12]
        pose = utils.homcoord_to_pose(label)
        return pose

    def getN_frames(self, scene, sequence):
        count = 0
        list_dir = os.listdir(os.path.join(self.scenes_dir, scene, sequence))
        for file in list_dir:
            if 'color' in str(file):
                count += 1
        return count

    def getStartIdx(self, scene_idx, sequence_idx):
        max_valid_start_index = (self.getN_frames(self.scenes_list[scene_idx],
                            self.get_sequences(self.scenes_list[scene_idx])[sequence_idx]) - self.frames_per_segment) // self.n_segment
        start_indices = np.multiply(list(range(self.n_segment)), max_valid_start_index) + \
                        np.random.randint(max_valid_start_index, size=self.n_segment)
        return start_indices

    def __len__(self):
        count = 0
        for i in range(self.n_scenes):
            sequences = self.get_sequences(self.scenes_list[i])
            count += len(sequences)
        return count

    def __getitem__(self, idx):
        frames = list()
        labels = list()
        l_frames = []
        temp = []
        count = 0
        scene_idx = 0
        sequence_idx = 0

        for i in range(self.n_scenes):
            sequences = self.get_sequences(self.scenes_list[i])
            count_p = count
            count += len(sequences)
            if idx < count:
                scene_idx = i
                sequence_idx = idx - count_p
                break

        if self.getN_frames(self.scenes_list[scene_idx],
                            self.get_sequences(self.scenes_list[scene_idx])[sequence_idx]) < self.n_segment * self.frames_per_segment:
            print("ERROR. The number of frames in dataset is smaller than the number required.")

        for start_idx in self.getStartIdx(scene_idx, sequence_idx):
            frame_idx = start_idx

            for i in range(self.frames_per_segment):
                f = self.load_frame(self.scenes_list[scene_idx],
                                    self.get_sequences(self.scenes_list[scene_idx])[sequence_idx], frame_idx)
                frames.append(f)
                # For each segment it takes the initial and last label
                if i == 0:
                    label1 = self.getLabel(self.scenes_list[scene_idx],
                                           self.get_sequences(self.scenes_list[scene_idx])[sequence_idx], frame_idx)
                if i == (self.frames_per_segment - 1):
                    label2 = self.getLabel(self.scenes_list[scene_idx],
                                           self.get_sequences(self.scenes_list[scene_idx])[sequence_idx], frame_idx)

                if frame_idx < self.getN_frames(self.scenes_list[scene_idx],
                                                self.get_sequences(self.scenes_list[scene_idx])[sequence_idx]) - 1:
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


class SevenScenesDataset_v2(SevenScenesDataset):
    """This dataloader takes n consecutive frames (overlapping last pose of previous item with the initial pose of the next item) each time until the frames in the current scene are finished.
     Then it passes to the next scene. """

    def __init__(self, scenes_dir, n_segment, frame_template, label_template, n_scenes, frames_per_segment, mode,
                 data_augmentation=False, transform=None):
        super().__init__(scenes_dir, n_segment, frame_template, label_template, n_scenes, frames_per_segment, mode,
                         data_augmentation, transform)
        self.scenes_dir = scenes_dir
        self.transform = transform
        self.frame_template = frame_template
        self.label_template = label_template
        self.n_scenes = n_scenes
        self.frames_per_segment = frames_per_segment
        self.n_frames = n_segment * frames_per_segment
        self.mode = mode
        self.scenes_list = self.get_scenes()
        self.data_augmentation = data_augmentation

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        count = 0
        for i in range(self.n_scenes):
            sequences = self.get_sequences(self.scenes_list[i])
            for sequence in sequences:
                count += self.getN_frames(self.scenes_list[i], sequence) // self.n_frames
        return count

    def __getitem__(self, idx):
        frames = list()
        labels = list()
        l_frames = []
        temp = []
        scene_index = self.n_scenes + 1
        sequence_index = 0
        count_p = 0
        count = 0

        for j in range(self.n_scenes):
            sequences = self.get_sequences(self.scenes_list[j])
            for z, sequence in enumerate(sequences):
                count_p = count
                count += self.getN_frames(self.scenes_list[j], sequence) // self.n_frames
                if idx < count:
                    sequence_index = z
                    break
            if idx < count:
                scene_index = j
                break

        if scene_index > self.n_scenes:
            raise Exception("Error Data-loader index.")

        if self.getN_frames(self.scenes_list[scene_index],
                            self.get_sequences(self.scenes_list[scene_index])[sequence_index]) < self.n_frames:
            raise Exception("ERROR. The number of frames in dataset is smaller than the number required.")

        frame_idx = (idx - count_p) * self.n_frames
        for i in range(self.n_frames):
            f = self.load_frame(self.scenes_list[scene_index],
                                self.get_sequences(self.scenes_list[scene_index])[sequence_index], (frame_idx + i))
            frames.append(f)
            # For each segment it takes the initial and last labels and calculate relative pose
            if (i + 1) % self.frames_per_segment == 0:
                label1 = self.getLabel(self.scenes_list[scene_index],
                                       self.get_sequences(self.scenes_list[scene_index])[sequence_index],
                                       (frame_idx + i + 1 - self.frames_per_segment))
                label2 = self.getLabel(self.scenes_list[scene_index],
                                       self.get_sequences(self.scenes_list[scene_index])[sequence_index],
                                       (frame_idx + i))
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


class SevenScenesDataset_overlapping(SevenScenesDataset):
    """This dataloader takes n consecutive frames (n/t poses) each time sliding of 1 frame
     until the frames in the current scene are finished. Then it passes to the next scene. """

    def __init__(self, scenes_dir, n_segment, frame_template, label_template, n_scenes, frames_per_segment, mode,
                 data_augmentation=False, transform=None):

        super().__init__(scenes_dir, n_segment, frame_template, label_template, n_scenes, frames_per_segment, mode,
                         data_augmentation, transform)
        self.img_dir = scenes_dir
        self.transform = transform
        self.frame_template = frame_template
        self.label_template = label_template
        self.n_scenes = n_scenes
        self.frames_per_segment = frames_per_segment
        self.n_frames = n_segment * frames_per_segment
        self.data_augmentation = data_augmentation
        self.mode = mode
        self.scenes_list = self.get_scenes()

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

    def __len__(self):
        count = 0
        for i in range(self.n_scenes):
            sequences = self.get_sequences(self.scenes_list[i])
            for sequence in sequences:
                count += self.getN_frames(self.scenes_list[i], sequence) // self.n_segment
        return count

    def __getitem__(self, idx):
        frames = list()
        labels = list()
        l_frames = []
        temp = []
        scene_index = self.n_scenes + 1
        count_p = 0
        count = 0

        for j in range(self.n_scenes):
            sequences = self.get_sequences(self.scenes_list[j])
            for z, sequence in enumerate(sequences):
                count_p = count
                count += self.getN_frames(self.scenes_list[j], sequence) // self.n_segment
                if idx < count:
                    sequence_index = z
                    break
            if idx < count:
                scene_index = j
                break

        if scene_index > self.n_scenes:
            print("Error Data-loader index.")
            return

        if self.getN_frames(self.scenes_list[scene_index],
                            self.get_sequences(self.scenes_list[scene_index])[sequence_index]) < self.n_frames:
            print("ERROR. The number of frames in dataset is smaller than the number required.")

        frame_idx = (idx - count_p)
        frame_idx *= self.n_segment
        for _ in range(self.n_segment):
            for j in reversed(range(self.frames_per_segment)):
                if frame_idx - j < 0:
                    f = self.load_frame(self.scenes_list[scene_index],
                                        self.get_sequences(self.scenes_list[scene_index])[sequence_index], 0)
                else:
                    f = self.load_frame(self.scenes_list[scene_index],
                                        self.get_sequences(self.scenes_list[scene_index])[sequence_index], (frame_idx - j))
                if j == 0:
                    if frame_idx - self.frames_per_segment < 0:
                        label1 = self.getLabel(self.scenes_list[scene_index],
                                               self.get_sequences(self.scenes_list[scene_index])[sequence_index], 0)
                    else:
                        label1 = self.getLabel(self.scenes_list[scene_index],
                                               self.get_sequences(self.scenes_list[scene_index])[sequence_index],
                                               (frame_idx + 1 - self.frames_per_segment))
                    label2 = self.getLabel(self.scenes_list[scene_index],
                                           self.get_sequences(self.scenes_list[scene_index])[sequence_index], (frame_idx - j))
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


