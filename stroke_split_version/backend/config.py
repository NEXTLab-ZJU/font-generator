import os
import torch
from app_util import *

current_work_dir = os.getcwd()

data_path = os.path.join(current_work_dir, f"./data")
alg_path = os.path.join(current_work_dir, f"./alg")
result_data_path = os.path.join(current_work_dir, f"./result_data")

makedir(result_data_path)

# # Is it an experimental version? Experimental versions will skip the training steps and directly read the content from pre-generated file paths.
# # If it's an experimental version, please refer to the readme and place the corresponding data in the "result_data/preview" directory.
is_preview = False

# # Is it a testing version? Testing versions will minimize the number of training epochs to quickly validate the pipeline.
is_test = True


current_step = 0
current_is_handling = False

charset = "阿"

zi2zi_train_epoch = 200
sd_finetune_steps = 15000

sd_infer_num = 6

default_zi2zi_result_path = ''
default_ti_data_path = ''
default_json_save_path = ''

current_vector_step = 0
current_vector_is_handling = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

segmentation_train_epoch = 20
segmentation_load_epoch = 20


if is_preview:
    default_zi2zi_result_path = os.path.join(current_work_dir, f"./result_data/preview/style_transfer/zi2zi_finetune")
    default_ti_data_path = os.path.join(current_work_dir, f"./result_data/preview/style_transfer/sd_finetune")
    default_json_save_path = os.path.join(current_work_dir, f"./result_data/preview/stroke_assemble/json_assemble")
    current_step = 5
    current_is_handling = False
    current_vector_is_handling = False
    current_vector_step = 6
    
if is_test:
    zi2zi_train_epoch = 1
    sd_finetune_steps = 3
    sd_infer_num = 1
    segmentation_train_epoch = 2
    segmentation_load_epoch = 2

