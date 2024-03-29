import torch
import yaml
import os
from torch.utils.data import DataLoader
import dataset_utils
from dataloader_rgbdwu import RGBDDataset_v2
from solver import Solver

CONFIG_PATH = "config/"


def config_load(config_filename):
    with open(os.path.join(CONFIG_PATH, config_filename)) as file:
        config = yaml.safe_load(file)

    return config


def get_mean_and_std_dataset(config):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    dataset = RGBDDataset_v2(config["dataset_path"], config["label_path"], config["n_segments"],
                             config["frame_template"], config["label_template"], config["n_video"],
                             config["f_per_segment"])
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
    for data, _ in dataloader:
        # Mean over batch, height and width, but not over the channels
        channels_sum += torch.mean(data, dim=[0, 1, 3, 4])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 1, 3, 4])
        num_batches += 1

    mean = channels_sum / num_batches

    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    return mean, std


def main():
    print("Insert config filename: ")
    config_filename = input()
    if(config_filename == ""):
        config = config_load("base_config")
    else:
        config = config_load(config_filename)
    solver = Solver(config)

    # Find dataset mean and standard deviation
    if config["data_init"]:
        print("Mean and standard deviation computation")
        m, s = get_mean_and_std_dataset(config)
        print("Mean: ", m)
        print("Std: ", s)
        return

    # Check config values
    assert config["n_segments"] * config["f_per_segment"] == config["n_frames"] and \
           config["patch_t"] == config["f_per_segment"]

    if config["mode"] == "train":
        # Load datasets if present otherwise create it
        data = dataset_utils.load_dataset(config)
        # Train the model
        solver.train(data)
    elif config["mode"] == "test":
        # Load dataset for testing
        data = dataset_utils.load_dataset(config)
        # Test the model
        solver.test(data)
    elif config["mode"] == "checkpoint":
        # Load dataset for testing
        data = dataset_utils.load_dataset(config)
        # Returns to the point the training stopped
        solver.train(data)
    else:
        raise Exception("ERROR. Mode not present")


if __name__ == '__main__':
    main()
