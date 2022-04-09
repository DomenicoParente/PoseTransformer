import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import OrderedDict
import model
from dataloader import RGBDDataset
import datetime
from datetime import datetime


class Solver:
    def __init__(self, config):
        # Device configuration
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print('GPU available')
        else:
            print('GPU not available')

        self.config = config
        self.model = model.PoseTransformer(self.config["n_frames"], self.config["height"], self.config["width"],
                                           self.config["patch_t"], self.config["patch_h"], self.config["patch_w"],
                                           self.config["channels"], self.config["dim_out"])

        # Create name for the current trained model
        now = datetime.now()
        self.model_name = "PT_model" + now.strftime("%d-%m-%H_%M")

        # Load pretrained model
        if self.config["pretrain"]:
            self.load_pretrained_model()

        self.model_save_path = 'trained_models/'

    def load_pretrained_model(self):
        model_path = self.config["pretrained_path"] + self.config["pretrained_model"]
        if torch.cuda.is_available():
            pretrained_model = torch.load(model_path)
        else:
            pretrained_model = torch.load(model_path, map_location=torch.device('cpu'))
        """
        if 'state_dict' in pretrained_model:
            pretrained_model = pretrained_model['state_dict']

        old_state_dict_keys = list(pretrained_model.keys())
        for old_key in old_state_dict_keys:
            print(old_key)
        """
        new_state_dict = OrderedDict()

        for key, value in pretrained_model.items():
            key = key[6:]  # remove `model.`
            new_state_dict[key] = value
        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
        # print(f'missing_keys:{missing_keys}\n unexpected_keys:{unexpected_keys}')
        print('Load pretrained network: ', model_path)

    def load_dataset(self, dataset_name):
        # Create the dataset and save files for training and testing.
        if not (os.path.isfile(dataset_name)):
            print('Setup dataset')

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.config["height"], self.config["width"])),
                transforms.Normalize(self.config["data_mean"], self.config["data_std"])
            ])

            dataset = RGBDDataset(self.config["dataset_path"], self.config["label_path"], self.config["n_segments"],
                                  self.config["frame_template"], self.config["label_template"], self.config["n_video"],
                                  self.config["f_per_segment"], transform)
            training_dataset = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=False)

            torch.save(training_dataset, dataset_name)
        train_data = torch.load(dataset_name)

        return train_data

    def train(self, train_data):
        print("Training starting")
        train_model = self.model.to(self.device)

        # Optimizer
        optimizer = optim.SGD(train_model.parameters(), lr=self.config["l_rate"])
        # Scheduler
        scheduler = StepLR(optimizer, step_size=self.config["n_epochs"], gamma=self.config["gamma"])

        # Creates a directory for the model when it does not exist
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        tot_loss = 0

        # Train NN for N epochs
        for epoch in range(self.config["epoch_start"], self.config["n_epochs"]):
            train_model.train()
            for batch_idx, (data, target) in enumerate(train_data):
                data = data.to(self.device)
                target = target.to(self.device)
                # print("Target:", target)
                # print('Size target: ', target.size())

                # Set to zero the parameter gradient
                optimizer.zero_grad()

                # Passing data to model
                output = train_model(data)
                # print('Size output: ', output.size())
                # print('Output:', output)
                loss = nn.L1Loss()(output, target)
                loss.backward()
                optimizer.step()

                tot_loss += loss.item()
                if batch_idx % self.config["log_interval"] == 0:
                    print('Training Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_data.dataset),
                               100. * batch_idx / len(train_data), tot_loss / (batch_idx + 1)))

            print('Number of epochs: ', epoch)
            print('Training Loss: {:.6f}'.format(tot_loss / len(train_data)))

            scheduler.step()

        # It saves the trained model
        if self.config["save"]:
            torch.save(train_model.state_dict(), os.path.join(self.model_save_path, self.model_name))

    def test(self, test_data):
        eval_model = self.model.to(self.device)
        eval_model.eval()
        tot_loss = 0
        correct = 0
        testing_loss = []
        testing_accuracy = []

        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(self.device), target.to(self.device)
                output = eval_model(data)
                tot_loss += torch.nn.CrossEntropyLoss()(output, target).item()  # sum up batch loss

        testing_loss.append(tot_loss / (len(test_data)))
        testing_accuracy.append(correct / (len(test_data) * self.config["batch_size"]))
        print('Test Loss: {:.6f}'.format(tot_loss / (len(test_data))))
