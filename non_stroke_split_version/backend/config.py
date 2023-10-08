import os
from util import *
import torch

# path config
current_work_dir = os.getcwd()
data_path = os.path.join(current_work_dir, f"./data")
alg_path = os.path.join(current_work_dir, f"./alg")
result_data_path = os.path.join(current_work_dir, f"./result_data")

makedir(data_path)
makedir(os.path.join(data_path,"./zips"))
makedir(os.path.join(data_path,"./upload_imgs"))
makedir(result_data_path)

# # Is it an experimental version? Experimental versions will skip the training steps and directly read the content from pre-generated file paths.
# # If it's an experimental version, please refer to the readme and place the corresponding data in the "result_data/preview" directory.
is_preview = True

# # Is it a testing version? Testing versions will minimize the number of training epochs to quickly validate the pipeline.
is_test = False

# The current step being executed.
current_step = 0
current_is_handling = False

# The characters needed for inference.
charset = "廒霸镑煸埔潺帱赐幌阳胤璎猷缀缒"

# zi2zi train epoch
zi2zi_train_epoch = 200

# sd finetune steps
sd_finetune_steps = 15000

# sd infer num
sd_infer_num = 6

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_zi2zi_result_path = ''
default_ti_data_path = ''


if is_preview:
    default_zi2zi_result_path = os.path.join(current_work_dir, f"./result_data/preview/style_transfer/zi2zi_finetune")
    default_ti_data_path = os.path.join(current_work_dir, f"./result_data/preview/style_transfer/sd_finetune")
    current_step = 5
    sd_infer_num = 6
    current_is_handling = False
    
if is_test:
    zi2zi_train_epoch = 1
    sd_finetune_steps = 3
    sd_infer_num = 6