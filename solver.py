import os
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import OrderedDict
import model
from dataloader import RGBDDataset, RGBDDatasetv2, RGBDDataset_test
import datetime
from datetime import datetime
from modules import PoseLoss
from matplotlib import pyplot as plt
import utils


class Solver:
    def __init__(self, config):
        # Device configuration
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print('GPU available')
        else:
            print('GPU not available')

        # Setting model, loss function
        self.config = config
        self.model = model.PoseTransformer(self.config["n_frames"], self.config["height"], self.config["width"],
                                           self.config["patch_t"], self.config["patch_h"], self.config["patch_w"],
                                           self.config["channels"], self.config["dim_out"])

        self.criterion = PoseLoss(self.device)

        # Create name for the current trained model
        now = datetime.now()
        self.model_name = "PT_model" + now.strftime("%d-%m-%H_%M")
        self.summary_name = "PT_model_summary" + now.strftime("%d-%m-%H_%M")
        self.loss_plot_name = "PT_model_loss_plot" + now.strftime("%d-%m-%H_%M")
        self.trajectory_plot_name = "PT_model_trajectory_plot" + now.strftime("%d-%m-%H_%M")

        # Load pretrained model
        if self.config["pretrain"] and self.config["mode"] == "train":
            self.load_pretrained_model()

        # Load model for testing
        if self.config["trained"] and self.config["mode"] == "test":
            self.load_trained_model()

        self.models_save_path = 'trained_models/'

    def load_pretrained_model(self):
        """ Load the pretrained model of ViViT"""
        model_path = self.config["pretrained_path"] + self.config["pretrained_model"]
        print("Setup pretrained model")
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

    def load_trained_model(self):
        """Load previous trained models for testing"""
        model_path = self.config["trained_path"] + self.config["trained_model"] + "/" + self.config["trained_model"]
        print("Setup trained model")
        if torch.cuda.is_available():
            pretrained_model = torch.load(model_path)
        else:
            pretrained_model = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(torch.load(model_path))

    def load_dataset(self, dataset_name):
        """Load the dataset and save files for training"""
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

    def load_dataset_test(self, dataset_name):
        """Load the dataset and save files for testing"""
        if not (os.path.isfile(dataset_name)):
            print('Setup test dataset')

            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((self.config["height"], self.config["width"])),
                transforms.Normalize(self.config["data_mean"], self.config["data_std"])
            ])

            dataset = RGBDDataset_test(self.config["dataset_path"], self.config["label_path"], self.config["n_segments"],
                                       self.config["frame_template"], self.config["label_template"], self.config["n_video"],
                                       self.config["f_per_segment"], transform)
            test_dataset = DataLoader(dataset, batch_size=1, shuffle=False)

            torch.save(test_dataset, dataset_name)
        test_data = torch.load(dataset_name)

        return test_data

    def train(self, train_data):
        print("Training starting")
        train_model = self.model.to(self.device)

        # Optimizer
        optimizer = optim.SGD(train_model.parameters(), lr=self.config["l_rate"])
        # Scheduler
        scheduler = StepLR(optimizer, step_size=self.config["n_epochs"], gamma=self.config["gamma"])

        # Creates a directory for the trained models when it does not exist
        if not os.path.exists(self.models_save_path) and self.config["save"]:
            os.makedirs(self.models_save_path)

        # Create a directory for the current trained model
        trained_model_path = self.models_save_path + self.model_name + "/"
        os.makedirs(trained_model_path)

        # Write summary
        if self.config["summary"]:
            summary_filepath = trained_model_path + self.summary_name
            summary_file = open(summary_filepath, 'w')
            summary_file.write("Summary " + self.model_name + "\n")
            summary_file.write("\n Input size: " + "[" +
                               str(self.config["channels"]) + " " + str(self.config["n_frames"]) + " " +
                               str(self.config["height"]) + " " + str(self.config["width"]) + "]\n")

        total_loss_training = []
        pos_loss_training = []
        ori_loss_training = []

        # Train NN for N epochs
        for epoch in range(0, self.config["n_epochs"]):
            train_model.train()
            tot_loss = 0
            for batch_idx, (data, target) in enumerate(train_data):
                data = data.to(self.device)
                target = target.to(self.device)
                # print("Target:", target)
                # print('Size target: ', target.size())

                # Set to zero the parameter gradient
                optimizer.zero_grad()

                # Passing data to model
                pos_out, ori_out = train_model(data)
                # print('Size output: ', pos_out.size(), ori_out.size())
                # print('Output:', pos_out, ori_out)

                ori_true = target[:, :, :4]
                pos_true = target[:, :, 4:]

                ori_out = F.normalize(ori_out, p=2, dim=1)
                ori_true = F.normalize(ori_true, p=2, dim=1)

                loss, _, _ = self.criterion(pos_out, ori_out, pos_true, ori_true)
                loss_t = self.criterion.loss_print[0]
                loss_pos = self.criterion.loss_print[1]
                loss_ori = self.criterion.loss_print[2]
                loss.backward()
                optimizer.step()

                tot_loss += loss.item()
                if batch_idx % self.config["log_interval"] == 0:
                    print('Training Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        epoch + 1, batch_idx * len(data), len(train_data.dataset),
                               100. * batch_idx / len(train_data), tot_loss / (batch_idx + 1)))

            total_loss_training.append(tot_loss/(len(train_data)))
            pos_loss_training.append(loss_pos)
            ori_loss_training.append(loss_ori)
            print('Number of epochs: ', epoch + 1)
            print('Training Total Loss: {:.6f} \t Training Position Loss: {:.6f} \t Training Orientation Loss: {:.6f}'.format(
                    tot_loss/(len(train_data)), loss_pos, loss_ori))

            if self.config["summary"]:
                n_ep = epoch + 1
                summary_file.write("Epoch: " + str(n_ep) + "\n")
                summary_file.write("Loss: " + str(loss_t) + "\n")
                summary_file.write("Position Loss: " + str(loss_pos) + "\n")
                summary_file.write("Orientation Loss: " + str(loss_pos) + "\n" + "\n")

            scheduler.step()

        print("Overall average position error {:.6f}".format(np.mean(pos_loss_training)))
        print("Overall average orientation error {:.6f}".format(np.mean(ori_loss_training)))

        if self.config["summary"]:
            summary_file.write("Overall average position error: " + str(np.mean(pos_loss_training)) + "\n")
            summary_file.write("Overall average orientation error: " + str(np.mean(ori_loss_training)) + "\n")

        if self.config["loss_plot"]:
            plot_path = trained_model_path + self.loss_plot_name
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.plot(range(self.config["n_epochs"]), total_loss_training, c='red')
            plt.ylabel('Loss Training')
            plt.xlabel('Number of Epochs')
            plt.savefig(plot_path)

        # It saves the trained model
        if self.config["save"]:
            torch.save(train_model.state_dict(), os.path.join(trained_model_path, self.model_name))

    def test(self, test_data):
        print("Starting Test")
        eval_model = self.model.to(self.device)
        eval_model.eval()
        target_pos = []
        target_ori = []
        estimated_pos = []
        estimated_ori = []
        pos_loss_testing = []
        ori_loss_testing = []
        t_x = []
        t_y = []
        t_z = []
        e_x = []
        e_y = []
        e_z = []
        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(self.device), target.to(self.device)
                pos_out, ori_out = eval_model(data)
                # print('Size output: ', pos_out.size(), ori_out.size())
                # print('Output:', pos_out, ori_out)
                pos_out = pos_out.squeeze(0).detach().cpu().numpy()
                ori_out = F.normalize(ori_out, p=2, dim=1)
                ori_out = ori_out.squeeze(0).detach().cpu().numpy()

                ori_true = target[:, :, :4].squeeze(0).numpy()
                pos_true = target[:, :, 4:].squeeze(0).numpy()

                #ori_true = utils.quat_to_euler(ori_true)
                #ori_out = utils.quat_to_euler(ori_out)

                loss_pos = utils.array_dist(pos_out, pos_true)
                loss_ori = utils.array_dist(ori_out, ori_true)
                pos_loss_testing.append(loss_pos)
                ori_loss_testing.append(loss_ori)

                for i in range(ori_true.shape[0]):
                    target_pos.append(pos_true[i])
                    target_ori.append(ori_true[i])
                    estimated_pos.append(pos_out[i])
                    estimated_ori.append(ori_out[i])

                for pos in target_pos:
                    t_x.append(pos[0])
                    t_y.append(pos[1])
                    t_z.append(pos[2])

                for pos in estimated_pos:
                    e_x.append(pos[0])
                    e_y.append(pos[1])
                    e_z.append(pos[2])

        print('Test Position Loss: {:.6f} \t Test Orientation Loss: {:.6f}'.format(
              np.mean(pos_loss_testing), np.mean(ori_loss_testing)))

        # Plot trajectory of using estimated poses and correct poses
        trajectory_plot_path = self.models_save_path + self.model_name + "/" + self.trajectory_plot_name
        utils.trajectory_plot(trajectory_plot_path, t_x, t_y, t_z, e_x, e_y, e_z)
