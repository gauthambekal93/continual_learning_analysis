# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 12:21:28 2026

@author: gauthambekal93
"""

import json
import random
import numpy as np
import torch
from torch import nn
import pandas as pd
import os
from tqdm import tqdm
import torch.nn.functional as F
from pathlib import Path
import shutil
import sys


experiment_dir = Path(__file__).resolve().parent
root_dir = Path(__file__).resolve().parent.parent 
sys.path.append(str(root_dir))

from utils.save_and_load import update_tracker

from datetime import datetime
import uuid


from models import resnet
from task_loop import run_tasks

def get_configs():
    config_names, configs = [], []
    config_dir =  os.path.join(root_dir ,"config_files" ) 
    
    for file_name in os.listdir(config_dir):
        if '.json' in file_name:
            file_path = os.path.join(config_dir, file_name)
            
            with open(  file_path , "r") as f:
                config = json.load(f)
            configs.append(config)    
            config_names.append(file_name)
    
    return configs, config_names

# ---------------- seeds ----------------
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_unique_id():
    run_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + "_"
        + str(uuid.uuid4())[:8]
    )

    return run_id


def move_config(config_name):
    src = os.path.join(root_dir ,"config_files" , config_name) 
    dst = os.path.join(root_dir ,"config_files" , "done_configs", config_name) 
    shutil.move(src, dst)


configs, config_names = get_configs()

for config_name, config in zip( config_names, configs ):
    
    set_seed( config["seed"])
    
    run_id = get_unique_id()

    update_tracker(root_dir, run_id, config)
    
    seed = config["seed"]
    depth = config["depth"]
    width = config["width"]
    num_tasks = config["num_tasks"]
    epochs_per_task = config["epochs_per_task"]
    batch_size = config["batch_size"]
    n_train = config["n_train"]
    n_test = config["n_test"]
    noise = config["noise"]
    lr= config["lr"]
    wd = config["wd"]
    activation = config["activation"]
    device = config["device"]
    out_dir = config["out_dir"]
    save_every = config["save_every"]
    print_every = config["print_every"]
    residual = config["residual"]
    norm = config["norm"]
    dropout = config["dropout"]
    opt_name = config["opt_name"]
    momentum = config["momentum"]
    return_pre_act = config["return_pre_act"]
    return_layer_out = config["return_layer_out"]
    
    
    run_tasks( 
     root_dir, 
     run_id,
     seed,    
     depth,
     width,
     num_tasks,
     epochs_per_task,
     batch_size,
     n_train,
     n_test,
     noise,
     lr,
     wd,
     activation,
     device,
     out_dir,
     save_every,
     print_every,
     residual ,
     norm,
     dropout,
     opt_name,
     momentum,
     return_pre_act,
     return_layer_out
     )
    
    move_config(config_name)
    update_tracker(root_dir, run_id, config)
    
        
