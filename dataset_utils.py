from torch.utils.data import DataLoader
from torchvision import transforms
import dataloader_rgbdwu as dtrgbdwu
import dataloader_7scenes as dtsc
import torch
import os

def transformations(config):
    if config["channels"] != 1:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config["height"], config["width"])),
            transforms.Normalize(config["data_mean"], config["data_std"])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((config["height"], config["width"])),
            transforms.Grayscale(),
            transforms.Normalize(config["data_mean"], config["data_std"])
        ])
    return transform

def load_dataset(config):
    """Load the dataset and save files for training"""
    if os.path.isfile(config["dataset_name"]):
        os.remove(config["dataset_name"])
    print('Setup dataset')

    # Basic transformations for all frames uploaded
    transform = transformations(config)

    if config["mode"] == "train":
        if config["dataset"] == "rgbd_wu":
            dataset = load_rgbdwu_dataset(config, transform)
        elif config["dataset"] == "7_scenes":
            dataset = load_7scenes_dataset(config, transform)
        else:
            raise Exception("Invalid dataset selected.")

    elif config["mode"] == "test":
        if config["dataset"] == "rgbd_wu":
            dataset = load_rgbdwu_dataset_test(config, transform)
        elif config["dataset"] == "7_scenes":
            dataset = load_7scenes_dataset(config, transform)
        else:
            raise Exception("Invalid dataset selected.")
    else:
        raise Exception("Invalid mode selected.")

    dataset_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                                  num_workers=config["n_workers"])
    torch.save(dataset_loader, config["dataset_name"])
    data = torch.load(config["dataset_name"])

    return data


def load_rgbdwu_dataset(config, transform):
    if config["dataloader"] == "v1":
        dataset = dtrgbdwu.RGBDDataset(config["dataset_path"], config["label_path"], config["n_segments"],
                                 config["frame_template"], config["label_template"],
                                 config["n_video"], config["f_per_segment"],
                                 config["data_augmentation"], transform)
    elif config["dataloader"] == "v2":
        dataset = dtrgbdwu.RGBDDataset_v2(config["dataset_path"], config["label_path"], config["n_segments"],
                                    config["frame_template"], config["label_template"], config["n_video"],
                                    config["f_per_segment"], config["data_augmentation"], transform)
    elif config["dataloader"] == "overlapping":
        dataset = dtrgbdwu.RGBDDataset_overlapping(config["dataset_path"], config["label_path"], config["n_segments"],
                                             config["frame_template"], config["label_template"], config["n_video"],
                                             config["f_per_segment"], config["data_augmentation"], transform)
    else:
        raise Exception("ERROR: Dataloader chosen is not valid")

    return dataset


def load_rgbdwu_dataset_test(config, transform):
    """Load the dataset and save files for testing"""
    if config["dataloader"] == "overlapping":
        dataset = dtrgbdwu.RGBDDataset_test_overlapping(config["dataset_path"], config["label_path"], config["n_segments"],
                                                        config["frame_template"], config["label_template"], config["n_video"],
                                                        config["f_per_segment"], config["test_scene"], transform)
    elif config["dataloader"] == "v2" or config["dataloader"] == "v1":
        dataset = dtrgbdwu.RGBDDataset_test(config["dataset_path"], config["label_path"], config["n_segments"],
                                            config["frame_template"], config["label_template"], config["n_video"],
                                            config["f_per_segment"], config["test_scene"], transform)
    else:
        raise Exception("ERROR: Dataloader chosen is not valid")

    return dataset


def load_7scenes_dataset(config, transform):
    if config["dataloader"] == "v1":
        dataset = dtsc.SevenScenesDataset(frame_template=config["frame_template"], scenes_dir=config["dataset_path"],
                                          label_template=config["label_template"], n_scenes=config["n_video"], mode=config["mode"],
                                          n_segment=config["n_segments"], frames_per_segment=config["f_per_segment"],
                                          data_augmentation=config["data_augmentation"], transform=transform)
    elif config["dataloader"] == "v2":
        dataset = dtsc.SevenScenesDataset_v2(frame_template=config["frame_template"], scenes_dir=config["dataset_path"],
                                          label_template=config["label_template"], n_scenes=config["n_video"],mode=config["mode"],
                                          n_segment=config["n_segments"], frames_per_segment=config["f_per_segment"],
                                          data_augmentation=config["data_augmentation"], transform=transform)
    elif config["dataloader"] == "overlapping":
        dataset = dtsc.SevenScenesDataset_overlapping(frame_template=config["frame_template"], scenes_dir=config["dataset_path"],
                                          label_template=config["label_template"], n_scenes=config["n_video"], mode=config["mode"],
                                          n_segment=config["n_segments"], frames_per_segment=config["f_per_segment"],
                                          data_augmentation=config["data_augmentation"], transform=transform)
    else:
        raise Exception("ERROR: Dataloader chosen is not valid")

    return dataset
