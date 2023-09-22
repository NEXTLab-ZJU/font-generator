import os
from util import *
import torch

# 路径配置
current_work_dir = os.getcwd()
data_path = os.path.join(current_work_dir, f"./data")
alg_path = os.path.join(current_work_dir, f"./alg")
result_data_path = os.path.join(current_work_dir, f"./result_data")

makedir(data_path)
makedir(os.path.join(data_path,"./zips"))
makedir(os.path.join(data_path,"./upload_imgs"))
makedir(result_data_path)

# # 是否实验版本,实验版本会跳过训练步骤，直接读取已经生成好的文件路径内容
# # 如果是实验版本，请根据readme把对应的数据放至 result_data/preview
is_preview = True

# # 是否是测试版本,测试版本会把训练轮数最小化，以快速验证通路
is_test = False

# 当前正在执行的步骤
current_step = 0
current_is_handling = False

# 需要生产推理的字符
charset = "廒霸镑煸埔潺帱赐幌阳胤璎猷缀缒"

# zi2zi训练轮数
zi2zi_train_epoch = 200

# sd微调轮数
sd_finetune_steps = 15000

# 对于每个字sd推理数量
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