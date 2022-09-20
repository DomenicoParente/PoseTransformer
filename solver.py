import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from collections import OrderedDict
import yaml
import model
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
        self.model = model.PoseTransformer(num_frames=self.config["n_frames"],
                                           height=self.config["height"],
                                           width=self.config["width"],
                                           patch_time=self.config["patch_t"],
                                           patch_height=self.config["patch_h"],
                                           patch_width=self.config["patch_w"],
                                           channels=self.config["channels"],
                                           dim_out=self.config["dim_out"],
                                           ldrop=self.config["last_dropout"])

        # Loss function selection
        self.criterion = PoseLoss(device=self.device,
                                  lossf=self.config["loss_function"],
                                  learn_beta=self.config["learn_beta"],
                                  sq=self.config["sq"])

        # Create name for the current trained model
        now = datetime.now()
        if self.config["mode"] != "checkpoint":
            self.model_name = "PT_model" + now.strftime("%m-%d-%H_%M")
        else:
            self.model_name = self.config["checkpoint_model"]

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
        self.model.load_state_dict(torch.load(model_path))

    def train(self, train_data):
        print("Training starting")
        print("Model: ", self.model_name)
        train_model = self.model.to(self.device)
        start = 0

        # Optimizer selection
        optimizer = utils.optimizer_selection(config=self.config,
                                              train_model=train_model,
                                              sx=self.criterion.sx,
                                              sq=self.criterion.sq)

        # Scheduler selection
        scheduler = utils.scheduler_selection(config=self.config,
                                              optimizer=optimizer)

        trained_model_path = self.models_save_path + self.model_name + "/"
        if self.config["mode"] == "train":
            # Creates a directory for the trained models when it does not exist
            if not os.path.exists(self.models_save_path) and self.config["save"]:
                os.makedirs(self.models_save_path)

            # Create a directory for the current trained model when it does not exist
            if not os.path.exists(trained_model_path):
                os.makedirs(trained_model_path)

            # Create a directory to store the training checkpoints
            if self.config["checkpoint"]:
                checkpoint_model_path = trained_model_path + "checkpoints/"
                if not os.path.exists(checkpoint_model_path):
                    os.makedirs(checkpoint_model_path)

        # it restarts the training from the epoch saved in the checkpoint
        if self.config["mode"] == "checkpoint":
            print("Checkpoint: ", self.config["checkpoint_to_load"])
            model_path = self.models_save_path + self.config["checkpoint_model"] + "/"
            checkpoint = torch.load(model_path + "checkpoints/" + self.config["checkpoint_to_load"])
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start = checkpoint['epoch']
            if self.config["summary"]:
                summary_filepath = self.models_save_path + self.config["checkpoint_model"] + "/" + \
                                   self.config["checkpoint_model"] + "_summary.txt"
                summary_file = open(summary_filepath, 'a')

        # Save config file in model directory
        config_file = trained_model_path + "config_file_" + self.model_name + ".yaml"
        file = open(config_file, 'w')
        yaml.dump(self.config, file, default_flow_style=False)

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
                if self.config["optimizer"] == "sgd":
                    summary_file.write("\n Momentum: " + str(self.config["momentum"]))
                if self.config["scheduler"] == "step":
                    summary_file.write("\n Step size: " + str(self.config["step_size"]))
                summary_file.write("\n Gamma: " + str(self.config["gamma"]))
                summary_file.write("\n Sq: " + str(self.config["sq"]))
                summary_file.write("\n Number of videos used for training: " + str(self.config["n_video"]))
                summary_file.write("\n Dataloader used: " + str(self.config["dataloader"]))
                summary_file.write("\n Loss function used: " + str(self.config["loss_function"]))
                summary_file.write("\n Optimizer used: " + str(self.config["optimizer"]))
                summary_file.write("\n Scheduler used: " + str(self.config["scheduler"]))
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
            start_time = time.time()

            for batch_idx, (data, target) in enumerate(train_data):
                data = data.to(self.device)
                target = target.to(self.device)
                # print("Target:", target)
                # print('Size target: ', target.size())

                # Set to zero the parameter gradient
                optimizer.zero_grad()

                pos_out, ori_out = train_model(data)
                # print('Size output: ', pos_out.size(), ori_out.size())
                # print('Output:', pos_out, ori_out)

                # 7-values pose: position + orientation
                pos_true = target[:, :, :3]
                ori_true = target[:, :, 3:]

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
            print('Time: ', time.time() - start_time)
            print('Training Total Loss: {:.6f} \t Training Position Loss: {:.6f} \t Training Orientation Loss: {:.6f}'.format(
                    tot_loss/(len(train_data)), loss_pos, loss_ori))

            if self.config["checkpoint"] and epoch % self.config["ep_checkpoint"] == 0 and epoch != 0:
                checkpoint_name = self.model_name + "_ep_" + str(epoch+1) + "_checkpoint.pt"
                checkpoint_path = self.models_save_path + self.model_name + "/" + "checkpoints/" + checkpoint_name
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': train_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, checkpoint_path)
                print("Checkpoint saved")

            if self.config["summary"]:
                n_ep = epoch + 1
                summary_file.write("Epoch: " + str(n_ep) + "\n")
                summary_file.write("Time:: " + str(time.time() - start_time) + "\n")
                summary_file.write("Loss: " + str(loss_t) + "\n")
                summary_file.write("Position Loss: " + str(loss_pos) + "\n")
                summary_file.write("Orientation Loss: " + str(loss_ori) + "\n" + "\n")

            scheduler.step()

        print("Overall average position loss {:.6f}".format(np.mean(pos_loss_training)))
        print("Overall average orientation loss {:.6f}".format(np.mean(ori_loss_training)))

        if self.config["summary"]:
            summary_file.write("Overall average position loss: " + str(np.mean(pos_loss_training)) + "\n")
            summary_file.write("Overall average orientation loss: " + str(np.mean(ori_loss_training)) + "\n")

        if self.config["loss_plot"] and self.config["mode"] == "train":
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
        target_poses = []
        estimated_poses = []
        eul_target_poses = []
        eul_estimated_poses = []
        pos_loss_testing = []
        ori_loss_testing = []

        # Create a directory for the current test
        test_path = self.models_save_path + self.config["trained_model"] + "/test_scene_" + str(self.config["test_scene"]) + "/"
        if not os.path.exists(test_path):
            os.makedirs(test_path)

        with torch.no_grad():
            for data, target in test_data:
                data, target = data.to(self.device), target.to(self.device)
                pos_out, ori_out = eval_model(data)
                # print('Size output: ', pos_out.size(), ori_out.size())
                # print('Output:', pos_out, ori_out)

                pos_out = pos_out.squeeze(0).detach().cpu().numpy()
                ori_out = ori_out.squeeze(0).detach().cpu().numpy()
                estimated_poses.append(np.concatenate((pos_out, ori_out), axis=1).squeeze(0))

                pose_true = target.squeeze(0).detach().cpu().numpy()
                target_poses.append(pose_true.squeeze(0))

        estimated_poses = utils.rel_to_glob(estimated_poses)
        target_poses = utils.rel_to_glob(target_poses)

        c_ori_true = np.zeros((target_poses.shape[0], 3))
        c_ori_out = np.zeros((estimated_poses.shape[0], 3))
        for c, q_true in enumerate(target_poses):
            c_ori_true[c] = utils.quat_to_euler(q_true[3:])
        for c, q_out in enumerate(estimated_poses):
            c_ori_out[c] = utils.quat_to_euler(q_out[3:])

        for i in range(target_poses.shape[0]):
            loss_pos = utils.cal_dist(estimated_poses[i][:3], target_poses[i][:3])
            pos_loss_testing.append(loss_pos)

        for i in range(c_ori_true.shape[0]):
            loss_ori = utils.cal_ori_err(c_ori_out[i], c_ori_true[i])
            ori_loss_testing.append(loss_ori)

        for i in range(target_poses.shape[0]):
            eul_target_poses.append(np.hstack((target_poses[i][:3], c_ori_true[i])))
            eul_estimated_poses.append(np.hstack((estimated_poses[i][:3], c_ori_out[i])))

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
                               ' Yaw: ' + str(ori_mean_loss[2]) + "\n")

        if self.config["cvs_file"]:
            path_target = test_path + self.config["trained_model"] + '_pose_target.csv'
            path_estim = test_path + self.config["trained_model"] + '_pose_estimation.csv'
            f1 = open(path_target, 'w')
            f2 = open(path_estim, 'w')
            writer1 = csv.writer(f1)
            writer2 = csv.writer(f2)
            writer1.writerows(target_poses)
            writer2.writerows(estimated_poses)

        if self.config["plot"]:
            target_pose = np.array(target_poses)
            estimated_pose = np.array(estimated_poses)
            nptarget_path = test_path + self.config["trained_model"] + '_tar.npy'
            npestimated_path = test_path + self.config["trained_model"] + '_est.npy'
            np.save(nptarget_path, target_pose)
            np.save(npestimated_path, estimated_pose)

            # Plot trajectory of using estimated poses and correct poses
            trajectory_plot_path = test_path + self.config["trained_model"] + "_trajectory_plot"
            utils.trajectory_plot(trajectory_plot_path, nptarget_path, npestimated_path)

            #Plot orientation
            orientation_plot_path = test_path + self.config["trained_model"] + "_orientation_plot"
            utils.orientation_plot(orientation_plot_path, nptarget_path, npestimated_path)

            #Plot errors graph
            pos_err_plot_path = test_path + self.config["trained_model"] + "_pos_err_plot"
            utils.position_err_plot(pos_err_plot_path, nptarget_path, npestimated_path)

            ori_err_plot_path = test_path + self.config["trained_model"] + "_ori_err_plot"
            utils.orientation_err_plot(ori_err_plot_path, nptarget_path, npestimated_path)
