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
from dataloader import RGBDDataset, RGBDDataset_v2, RGBDDataset_test
import datetime
from datetime import datetime
from modules import PoseLoss
import utils
from os.path import exists
import csv


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
        """
        self.model = model.PoseTransformer_v1(self.config["n_frames"], self.config["height"], self.config["width"],
                                           self.config["patch_t"], self.config["patch_h"], self.config["patch_w"],
                                           self.config["channels"], self.config["dim"], self.config["n_heads"],
                                           self.config["mlp_dim"], self.config["depth"], self.config["dim_out"],
                                           self.config["last_dropout"])
        """

        self.model = model.PoseTransformer(self.config["n_frames"], self.config["height"], self.config["width"],
                                           self.config["patch_t"], self.config["patch_h"], self.config["patch_w"],
                                           self.config["channels"], self.config["dim_out"], self.config["last_dropout"])

        self.criterion = PoseLoss(self.device, self.config["beta"])

        # Create name for the current trained model
        now = datetime.now()
        self.model_name = "PT_model" + now.strftime("%m-%d-%H_%M")

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
            key = key[6:]
            new_state_dict[key] = value
        missing_keys, unexpected_keys = self.model.load_state_dict(new_state_dict, strict=False)
        # print(f'missing_keys:{missing_keys}\n unexpected_keys:{unexpected_keys}')
        print('Load pretrained network: ', model_path)

    def load_trained_model(self):
        """Load previous trained models for testing"""
        model_path = self.config["trained_path"] + self.config["trained_model"] + "/" + self.config["trained_model"] + ".pth"
        print("Setup trained model")
        if torch.cuda.is_available():
            pretrained_model = torch.load(model_path, map_location=torch.device('cpu'))
        else:
            pretrained_model = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(torch.load(model_path))

    def load_dataset(self, dataset_name):
        """Load the dataset and save files for training"""
        if os.path.isfile(dataset_name):
            os.remove(dataset_name)
        print('Setup dataset')

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.config["height"], self.config["width"])),
            transforms.Normalize(self.config["data_mean"], self.config["data_std"])
        ])
        """
        dataset = RGBDDataset(self.config["dataset_path"], self.config["label_path"], self.config["n_segments"],
                                self.config["frame_template"], self.config["label_template"],
                                self.config["n_video"], self.config["f_per_segment"],
                                self.config["data_augmentation"], transform)
        """
        dataset = RGBDDataset_v2(self.config["dataset_path"], self.config["label_path"], self.config["n_segments"],
                                 self.config["frame_template"], self.config["label_template"], self.config["n_video"],
                                 self.config["f_per_segment"], self.config["data_augmentation"], transform)

        training_dataset = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=False,
                                      num_workers=self.config["n_workers"])
        torch.save(training_dataset, dataset_name)
        train_data = torch.load(dataset_name)

        return train_data

    def load_dataset_test(self, dataset_name):
        """Load the dataset and save files for testing"""
        if os.path.isfile(dataset_name):
            os.remove(dataset_name)
        print('Setup test dataset')

        # Basic transformations for all frames uploaded
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((self.config["height"], self.config["width"])),
            transforms.Normalize(self.config["data_mean"], self.config["data_std"])
        ])

        dataset = RGBDDataset_test(self.config["dataset_path"], self.config["label_path"], self.config["n_segments"],
                                   self.config["frame_template"], self.config["label_template"], self.config["n_video"],
                                   self.config["f_per_segment"], self.config["test_scene"], transform)
        test_dataset = DataLoader(dataset, batch_size=1, shuffle=False)

        torch.save(test_dataset, dataset_name)
        test_data = torch.load(dataset_name)

        return test_data


    def train(self, train_data):
        print("Training starting")
        print("Model: ", self.model_name)
        train_model = self.model.to(self.device)
        start = 0

        # Optimizer
        optimizer = optim.SGD(train_model.parameters(), lr=self.config["l_rate"], momentum=self.config["momentum"],
                              weight_decay=self.config["weight_decay"])
        # Scheduler
        scheduler = StepLR(optimizer, step_size=self.config["step_size"], gamma=self.config["gamma"])

        if self.config["mode"] == "train":
            # Creates a directory for the trained models when it does not exist
            if not os.path.exists(self.models_save_path) and self.config["save"]:
                os.makedirs(self.models_save_path)

            # Create a directory for the current trained model when it does not exist
            trained_model_path = self.models_save_path + self.model_name + "/"
            if not os.path.exists(trained_model_path):
                os.makedirs(trained_model_path)

            # Create a directory to store the training checkpoints
            if self.config["checkpoint"]:
                checkpoint_model_path = trained_model_path + "checkpoints/"
                if not os.path.exists(checkpoint_model_path):
                    os.makedirs(checkpoint_model_path)

        # it restarts the training from the epoch saved in the checkpoint
        if self.config["mode"] == "checkpoint":
            print("Check point: ", self.config["checkpoint_to_load"])
            model_path = self.models_save_path + self.config["checkpoint_model"] + "/"
            checkpoint = torch.load(model_path + "checkpoints/" + self.config["checkpoint_to_load"])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start = checkpoint['epoch']
            if self.config["summary"]:
                summary_filepath = self.models_save_path + self.config["checkpoint_model"] + "/" + \
                                   self.config["checkpoint_model"] + "_summary.txt"
                summary_file = open(summary_filepath, 'a')

        # Write summary
        if start == 0:
            if self.config["summary"]:
                summary_filepath = trained_model_path + self.model_name + "_summary.txt"
                summary_file = open(summary_filepath, 'w')
                summary_file.write("Summary " + self.model_name + "\n")
                summary_file.write("\n Input size: " + "[" +
                                   str(self.config["channels"]) + " " + str(self.config["n_frames"]) + " " +
                                   str(self.config["height"]) + " " + str(self.config["width"]) + "]\n")
                summary_file.write("\n Patch size: " + "[" +
                                   str(self.config["patch_w"]) + " " + str(self.config["patch_h"]) + " " +
                                   str(self.config["patch_t"]) + "]\n")
                summary_file.write("\n Number of epochs: " + str(self.config["n_epochs"]))
                summary_file.write("\n Initial learning rate: " + str(self.config["l_rate"]))
                summary_file.write("\n Momentum: " + str(self.config["momentum"]))
                summary_file.write("\n Step size: " + str(self.config["step_size"]))
                summary_file.write("\n Gamma: " + str(self.config["gamma"]))
                summary_file.write("\n Beta: " + str(self.config["beta"]))
                if self.config["pretrain"]:
                    summary_file.write("\n The model is pretrained\n")
                else:
                    summary_file.write("\n The model is not pretrained\n")
                if self.config["data_augmentation"]:
                    summary_file.write("\n Data augmentation ON\n")
                else:
                    summary_file.write("\n Data augmentation OFF\n")
                summary_file.write("\n")
                summary_file.write(repr(self.model))
                summary_file.write("\n")


        total_loss_training = []
        pos_loss_training = []
        ori_loss_training = []

        # Train NN for N epochs
        for epoch in range(start, self.config["n_epochs"]):
            train_model.train()
            tot_loss = 0
            for batch_idx, (data, target) in enumerate(train_data):
                data = data.to(self.device)
                target = target.to(self.device)
                #print("Target:", target)
                # print('Size target: ', target.size())

                # Set to zero the parameter gradient
                optimizer.zero_grad()

                # Passing data to model
                pos_out, ori_out = train_model(data)
                # print('Size output: ', pos_out.size(), ori_out.size())
                # print('Output:', pos_out, ori_out)

                ori_true = target[:, :, :4]
                pos_true = target[:, :, 4:]

                ori_out = F.normalize(ori_out, p=2, dim=2)
                ori_true = F.normalize(ori_true, p=2, dim=2)

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
            print('Number of epoch: ', epoch + 1)
            print('Training Total Loss: {:.6f} \t Training Position Loss: {:.6f} \t Training Orientation Loss: {:.6f}'.format(
                    tot_loss/(len(train_data)), loss_pos, loss_ori))

            if self.config["checkpoint"] and epoch % self.config["ep_checkpoint"] == 0 and epoch != 0:
                checkpoint_name = self.model_name + "_ep_" + str(epoch+1) + "_checkpoint.pt"
                checkpoint_path = checkpoint_model_path + checkpoint_name
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': train_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)
                print("Checkpoint saved")

            if self.config["summary"]:
                n_ep = epoch + 1
                summary_file.write("Epoch: " + str(n_ep) + "\n")
                summary_file.write("Loss: " + str(loss_t) + "\n")
                summary_file.write("Position Loss: " + str(loss_pos) + "\n")
                summary_file.write("Orientation Loss: " + str(loss_ori) + "\n" + "\n")

            scheduler.step()

        print("Overall average position loss {:.6f}".format(np.mean(pos_loss_training)))
        print("Overall average orientation loss {:.6f}".format(np.mean(ori_loss_training)))

        if self.config["summary"]:
            summary_file.write("Overall average position loss: " + str(np.mean(pos_loss_training)) + "\n")
            summary_file.write("Overall average orientation loss: " + str(np.mean(ori_loss_training)) + "\n")

        if self.config["loss_plot"]:
            utils.loss_plot(trained_model_path, self.model_name, self.config["n_epochs"], total_loss_training)

        # It saves the trained model
        if self.config["save"]:
            trained_model_path = trained_model_path + self.model_name + ".pth"
            torch.save(train_model.state_dict(), trained_model_path)
            print("Model successfully saved")

    def test(self, test_data):
        print("Starting Test")
        eval_model = self.model.to(self.device)
        eval_model.eval()
        target_pose = []
        estimated_pose = []
        pos_loss_testing = []
        ori_loss_testing = []
        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(self.device), target.to(self.device)
                pos_out, ori_out = eval_model(data)
                # print('Size output: ', pos_out.size(), ori_out.size())
                # print('Output:', pos_out, ori_out)
                pos_out = pos_out.squeeze(0).detach().cpu().numpy()
                ori_out = F.normalize(ori_out, p=2, dim=2)
                ori_out = ori_out.squeeze(0).detach().cpu().numpy()

                ori_true = F.normalize(target[:, :, :4], p=2, dim=2).squeeze(0).detach().cpu().numpy()
                pos_true = target[:, :, 4:].squeeze(0).detach().cpu().numpy()

                c_ori_true = np.zeros((ori_true.shape[0], 3))
                c_ori_out = np.zeros((ori_out.shape[0], 3))
                for c, q_true in enumerate(ori_true):
                    c_ori_true[c] = utils.quat_to_euler(q_true)
                for c, q_out in enumerate(ori_out):
                    c_ori_out[c] = utils.quat_to_euler(q_out)

                for i in range(pos_true.shape[0]):
                    loss_pos = utils.cal_dist(pos_out[i], pos_true[i])
                    pos_loss_testing.append(loss_pos)

                for i in range(c_ori_true.shape[0]):
                    loss_ori = utils.cal_ori_err(c_ori_out[i], c_ori_true[i])
                    ori_loss_testing.append(loss_ori)

                for i in range(pos_true.shape[0]):
                    target_pose.append(np.hstack((pos_true[i], c_ori_true[i])))
                    estimated_pose.append(np.hstack((pos_out[i], c_ori_out[i])))

        ori_loss_testing = np.array(ori_loss_testing)
        ori_mean_loss = np.mean(ori_loss_testing, 0)

        print('Test Position Error: {:.6f} \n Test Orientation Error: \n Roll: {:.3f} Pitch: {:.3f}  Yaw: {:.3f}'.format(np.mean(pos_loss_testing), ori_mean_loss[0], ori_mean_loss[1], ori_mean_loss[2]))

        summary_filepath = self.models_save_path + self.config["trained_model"] + "/" + self.config["trained_model"] + "_summary.txt"
        if self.config["summary"] and exists(summary_filepath):
            summary_file = open(summary_filepath, 'a')
            summary_file.write("Overall test position error: " + str(np.mean(pos_loss_testing)) + "\n")
            summary_file.write("Overall test orientation error: " + "\n" +
                               'Roll: ' + str(ori_mean_loss[0]) +
                               ' Pitch: ' + str(ori_mean_loss[1]) +
                               ' Yaw: ' + str(ori_mean_loss[2]))

        if self.config["cvs_file"]:
            path_target = self.models_save_path + self.config["trained_model"] + "/" + self.config["trained_model"] + '_pose_target.csv'
            path_estim = self.models_save_path + self.config["trained_model"] + "/" + self.config["trained_model"] + '_pose_estimation.cvs'
            f1 = open(path_target, 'w')
            f2 = open(path_estim, 'w')
            writer1 = csv.writer(f1)
            writer2 = csv.writer(f2)
            writer1.writerows(target_pose)
            writer2.writerows(estimated_pose)

        if self.config["trajectory_plot"]:
            target_pose = np.array(target_pose)
            estimated_pose = np.array(estimated_pose)
            nptarget_path = self.models_save_path + self.config["trained_model"] + "/" + self.config[
                "trained_model"] + '_tar.npy'
            npestimated_path = self.models_save_path + self.config["trained_model"] + "/" + self.config[
                "trained_model"] + '_est.npy'
            np.save(nptarget_path, target_pose)
            np.save(npestimated_path, estimated_pose)

            # Plot trajectory of using estimated poses and correct poses
            trajectory_plot_path = self.models_save_path + self.config["trained_model"] + "/" + self.config["trained_model"] + "_trajectory_plot"
            utils.trajectory_plot(trajectory_plot_path, nptarget_path, npestimated_path)

            #Plot orientation
            orientation_plot_path = self.models_save_path + self.config["trained_model"] + "/" + self.config["trained_model"] + "_orientation_plot"
            utils.orientation_plot(orientation_plot_path, nptarget_path, npestimated_path)

            #Plot errors graph
            pos_err_plot_path = self.models_save_path + self.config["trained_model"] + "/" + self.config[
                "trained_model"] + "_pos_err_plot"
            utils.position_err_plot(pos_err_plot_path, nptarget_path, npestimated_path)

            ori_err_plot_path = self.models_save_path + self.config["trained_model"] + "/" + self.config[
                "trained_model"] + "_ori_err_plot"
            utils.orientation_err_plot(ori_err_plot_path, nptarget_path, npestimated_path)