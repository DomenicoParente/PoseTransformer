from torch.utils.data import DataLoader
from torchvision import transforms
import dataloader as dt
import torch
import os


def load_dataset(config):
    """Load the dataset and save files for training"""
    if os.path.isfile(config["dataset_name"]):
        os.remove(config["dataset_name"])
    print('Setup dataset')

    # Basic transformations for all frames uploaded
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
    if config["dataloader"] == "v1":
        dataset = dt.RGBDDataset(config["dataset_path"], config["label_path"], config["n_segments"],
                                 config["frame_template"], config["label_template"],
                                 config["n_video"], config["f_per_segment"],
                                config["data_augmentation"], transform)
    elif config["dataloader"] == "v2":
        dataset = dt.RGBDDataset_v2(config["dataset_path"], config["label_path"], config["n_segments"],
                                    config["frame_template"], config["label_template"], config["n_video"],
                                    config["f_per_segment"], config["data_augmentation"], transform)
    elif config["dataloader"] == "overlapping":
        dataset = dt.RGBDDataset_overlapping(config["dataset_path"], config["label_path"], config["n_segments"],
                                             config["frame_template"], config["label_template"], config["n_video"],
                                             config["f_per_segment"], config["data_augmentation"], transform)
    else:
        print("ERROR: Dataloader chosen is not valid")
        return

    training_dataset = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                                  num_workers=config["n_workers"])
    torch.save(training_dataset, config["dataset_name"])
    train_data = torch.load(config["dataset_name"])

    return train_data

def load_dataset_test(config):
    """Load the dataset and save files for testing"""
    if os.path.isfile(config["dataset_name"]):
        os.remove(config["dataset_name"])
    print('Setup test dataset')

    # Basic transformations for all frames uploaded
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

    if config["dataloader"] == "overlapping":
        dataset = dt.RGBDDataset_test_overlapping(config["dataset_path"], config["label_path"], config["n_segments"],
                                                  config["frame_template"], config["label_template"], config["n_video"],
                                                  config["f_per_segment"], config["test_scene"], transform)
    else:
        dataset = dt.RGBDDataset_test(config["dataset_path"], config["label_path"], config["n_segments"],
                                      config["frame_template"], config["label_template"], config["n_video"],
                                      config["f_per_segment"], config["test_scene"], transform)

    test_dataset = DataLoader(dataset, batch_size=1, shuffle=False)

    torch.save(test_dataset, config["dataset_name"])
    test_data = torch.load(config["dataset_name"])

    return test_data