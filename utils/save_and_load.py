# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 15:12:07 2026

@author: gauthambekal93
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 09:26:21 2026

@author: gauthambekal93
"""
import os
import pandas as pd
import torch
from tqdm import tqdm

def update_tracker(root_dir, run_id, config):
    
    tracker_path = os.path.join(
        root_dir,
        "tracker.csv"
    )
    
    if os.path.exists(tracker_path):
        
        df = pd.read_csv(tracker_path)
        
        temp = {"run_id": run_id}
        
        if run_id in df.run_id.values:
        
            df.loc[df['run_id'] == run_id, 'status'] = "completed"
            
        else:
            temp.update({"status":"not completed"})
        
            temp.update(config)
        
            new_row = pd.DataFrame([temp])
        
            df = pd.concat([df, new_row], ignore_index=True)
    
    else:
    
        df = new_row
    
    df.to_csv(tracker_path, index=False)


def save_checkpoint(net, opt, rows, root_dir, out_dir, run_id, task):
    
    directory = os.path.join(root_dir, out_dir, run_id) 
    
    os.makedirs( directory , exist_ok=True)

    df = pd.DataFrame(rows)
    
    csv_path = os.path.join( directory, run_id+"_results_logs.csv" )
    
    df.to_csv(csv_path, index=False)

    path = os.path.join( directory , run_id+'_model.pt' )

    torch.save({
        "task": task,
        "model_state_dict": net.state_dict(),
        "optimizer_state_dict": opt.state_dict()
    }, path)

    tqdm.write(f"Saved checkpoint at task {task}")


def load_checkpoint():
    pass