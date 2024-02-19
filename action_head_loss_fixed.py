#!/usr/bin/env python
# -*- coding: utf-8 -*-

" Train from the replay files through python tensor (pt) file"

import os
python_file_path= os.path.dirname(os.path.abspath(__file__))
print(python_file_path)
import gc

import sys
import time
import traceback
import argparse
import datetime

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, ConcatDataset, random_split
from torch.optim import Adam, RMSprop
from torch.optim.lr_scheduler import StepLR
import tensordict

from tensorboardX import SummaryWriter

from absl import flags
from absl import app
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt

from alphastarmini.core.arch.arch_model import ArchModel

from alphastarmini.core.sl.feature import Feature
from alphastarmini.core.sl.label import Label
from alphastarmini.core.sl import sl_loss_multi_gpu as Loss
from alphastarmini.core.sl.dataset import ReplayTensorDataset
from alphastarmini.core.sl import sl_utils as SU

from alphastarmini.lib.utils import load_latest_model, initial_model_state_dict

from alphastarmini.lib.hyper_parameters import Arch_Hyper_Parameters as AHP
from alphastarmini.lib.hyper_parameters import SL_Training_Hyper_Parameters as SLTHP
from alphastarmini.lib.hyper_parameters import StarCraft_Hyper_Parameters as SCHP

import param as P

__author__ = "Ruo-Ze Liu"

class CustomDataset:
    """
    Custom Dataset class which read data tuples from disk
    only when it is needed.

    :param file_location: Relative path to the files.
    :param file_num: The number of files.
    :param file_size: The number of data in one file.
    :param replay_files: The names of all the replay files.
    """
    def __init__(self, file_location, file_num, file_size, replay_files):
        super(CustomDataset, self).__init__()

        self.file_location = file_location
        self.file_num = file_num
        self.file_size = file_size
        self.replay_files = replay_files

    def __getitem__(self, idx):

        feature_lst = []
        label_lst = []

        for temp_idx in range(idx, idx+SEQ_LEN):
            replay_file = self.replay_files[temp_idx]
            replay_path = self.file_location + replay_file
            start_load = time.time()
            try:
                temp = torch.load(replay_path)
                feature, label = temp
            except RuntimeError:
                print(f"s,a file {replay_file} damaged.")
                return None
                feature, label = copy.deepcopy(feature), copy.deepcopy(label)
            except EOFError:
                print(f"s,a file {replay_file} is empty.")
                return None
                feature, label = copy.deepcopy(feature), copy.deepcopy(label)
            # print(f"single load time: {time.time() - start_load}, file: {replay_file}")
            feature_lst.extend([feature])
            label_lst.extend([label])
        
        feature_tensor = torch.concat(feature_lst)
        label_tensor = torch.concat(label_lst)
        
        return feature_tensor, label_tensor

    def __len__(self):
        return self.file_num * self.file_size

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

debug = False

parser = argparse.ArgumentParser()
parser.add_argument("-p1", "--path1", default="/sc2-dataset/RushBuild_tensor/", help="The path where data stored")
parser.add_argument("-p2", "--path2", default="./data/replay_data_tensor_new_small_AR/", help="The path where data stored")
parser.add_argument("-p3", "--path3", default="./data/replay_data_tensor_2019wcs_10m_step/", help="The path where data stored")
parser.add_argument("-m", "--model", choices=["sl", "rl"], default="sl", help="Choose model type")
parser.add_argument("-r", "--restore", action="store_true", default=False, help="whether to restore model or not")
parser.add_argument("-c", "--clip", action="store_true", default=False, help="whether to use clipping")
parser.add_argument('--num_workers', type=int, default=3, help='')


args = parser.parse_args()

# training paramerters
if SCHP.map_name == 'Simple64':
    PATH = args.path1
elif SCHP.map_name == 'AbyssalReef':
    PATH = args.path2
else:
    PATH = args.path3

#PATH = './temp/'

SAVE_STATE_DICT = True
MODEL_PATH = "./model/"
MODEL = "sl"
if not os.path.exists(MODEL_PATH):
    os.mkdir(MODEL_PATH)

print(PATH)
    
NUM_WORKERS = args.num_workers

SIMPLE_TEST = not P.on_server

# hyper paramerters
# use the same as in RL

# important: use larger batch_size and smaller seq_len in SL!
BATCH_SIZE = 64#3 * AHP.batch_size
SEQ_LEN = int(AHP.sequence_length * 0.5)

print('BATCH_SIZE:', BATCH_SIZE) if debug else None
print('SEQ_LEN:', SEQ_LEN) if debug else None

#REPLAY_RATIO = 0.6
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
STEP_SIZE = 50
GAMMA = 0.2

torch.manual_seed(SLTHP.seed)
np.random.seed(SLTHP.seed)

def main_worker(device):#, used, cnt):
    print('==> Making model..')
    net = ArchModel()
    # net.load_state_dict(torch.load("./model/sl_24-02-12_19-59-12.pth"))
    checkpoint = None
    
    net = net.to(device)

    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print('The number of parameters of model is', num_params)

    print('==> Making optimizer and scheduler..')

    optimizer = Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    print('==> Preparing data..')

    replay_files = os.listdir(PATH)

    import natsort
    
    print('length of replay_files:', len(replay_files)) if debug else None
    replay_files.sort()
    replay_files = natsort.natsorted(replay_files)

    dataset = CustomDataset(
        PATH, 
        len(replay_files)-SEQ_LEN,
        1,
        replay_files
        )
    
    dataset_size = len(dataset)
    train_size = int(dataset_size * 0.8)
    validation_size = dataset_size - train_size
#    test_size = dataset_size - train_size - validation_size

    train_set, valid_set = random_split(dataset, [train_size, validation_size])
   
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        collate_fn=collate_fn
        )
    
    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=False,
        collate_fn=collate_fn
        )
    
    return train(net, optimizer, scheduler, train_loader, valid_loader, device)

def train(net, optimizer, scheduler, train_loader, valid_loader, device):
    
    print('==> Preparing training..')
    
    time_str = time.strftime("%y-%m-%d_%H-%M-%S", time.localtime())
    SAVE_PATH = os.path.join(MODEL_PATH, MODEL + "_" + time_str)
    
    print('==> start training..')
    print('epoch:',NUM_EPOCHS)
    print('batch_size:',BATCH_SIZE)
    print('batch_num:',len(train_loader))

    start = time.time()
    for ep in range(0,NUM_EPOCHS):
        loss_sum = 0
        batch_time = time.time()

        print_num = 100
        # put model in train mode
        net.train()
        for batch_idx, train_batch in enumerate(train_loader):

            features, labels = train_batch
            print('features.shape:', features.shape) if batch_idx==0 else None
            print('labels.shape::', labels.shape) if batch_idx==0 else None
            # print('batch index: ', batch_idx)
            
            feature_tensor = features.to(device).float()
            labels_tensor = labels.to(device).float()
            del features, labels
            
            loss, loss_list, \
                acc_num_list = Loss.get_sl_loss_for_tensor(feature_tensor, 
                                                           labels_tensor, net, 
                                                           decrease_smart_opertaion=True,
                                                           return_important=True,
                                                           only_consider_small=False,
                                                           train=True)

            action_accuracy = acc_num_list[0] / (acc_num_list[1] + 1e-9)

            loss_sum += loss.item()
            
            optimizer.zero_grad()
            loss.backward() 
            optimizer.step()
            
            if batch_idx % print_num == print_num-1:
                print('Batch/Epoch: [{}/{}]| loss: {:.3f} | acc: {:.3f} | {} batch time: {:.3f}'.format(
                    batch_idx+1, ep+1, loss_sum/print_num, action_accuracy, print_num, time.time() - batch_time))
                loss_sum = 0
                batch_time = time.time()
                torch.cuda.empty_cache()
                
            
            gc.collect()
            
            
        if SAVE_STATE_DICT:
            save_path = SAVE_PATH + ".pth"
            print('Save model state_dict to', save_path)
            torch.save(net.state_dict(), save_path)
        
        SU.action_recall = dict()
        
        scheduler.step()
        
        print('########## validation ###########')
        
        print_num = 25
        loss_sum = 0
        batch_time = time.time()
        net.eval()
        with torch.no_grad():
            for batch_idx, valid_batch in enumerate(valid_loader):
                features, labels = valid_batch
                print('features.shape:', features.shape) if batch_idx==0 else None
                print('labels.shape::', labels.shape) if batch_idx==0 else None
                # print('batch index: ', batch_idx)
                
                feature_tensor = features.to(device).float()
                labels_tensor = labels.to(device).float()
                del features, labels
                
                loss, loss_list, \
                    acc_num_list = Loss.get_sl_loss_for_tensor(feature_tensor, 
                                                            labels_tensor, net, 
                                                            decrease_smart_opertaion=True,
                                                            return_important=True,
                                                            only_consider_small=False,
                                                            train=True)

                action_accuracy = acc_num_list[0] / (acc_num_list[1] + 1e-9)

                loss_sum += loss.item()
                
                action_recall = SU.action_recall
                
                if batch_idx % print_num == print_num-1:
                    print('Batch/Epoch: [{}/{}]| loss: {:.3f} | acc: {:.3f} | {} batch time: {:.3f}'.format(
                        batch_idx+1, ep+1, loss_sum/print_num, action_accuracy, print_num, time.time() - batch_time))
                    loss_sum = 0
                    batch_time = time.time()

            action_recall = SU.action_recall

            plt.figure()
            keys = action_recall.keys()
            recalls = [action_recall[key][0]/action_recall[key][1] for key in keys]
            x = np.arange(len(keys))
            plt.bar(x, recalls)
            plt.xticks(x, keys, rotation=90)
            plt.savefig(f"./outputs/action_recall_{ep+1}.png")

        
    print("training time:",time.time()-start)
    return

def test():

    # gpu setting
    ON_GPU = torch.cuda.is_available()
    print("cuda.is_available:",ON_GPU)
    DEVICE = torch.device("cuda:0" if ON_GPU else "cpu")
    print("device:",DEVICE.type)

    if ON_GPU:
        if torch.backends.cudnn.is_available():
            print('cudnn available')
            print('cudnn version', torch.backends.cudnn.version())
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
    
    return main_worker(DEVICE)

if __name__ == "__main__":
    test()
