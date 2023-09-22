
from flask import Flask, request,send_file
import os
import zipfile
import uuid
from config import *
from app_util import *
import sys



sys.path.append('.')
sys.path.append(os.path.join(alg_path,'./style_transfer'))
sys.path.append(os.path.join(alg_path,'./stroke_assemble'))

from alg.style_transfer.zi2zi_fintune import *
from alg.style_transfer.text_inversion_fintune import *

from alg.stroke_assemble.cal_moment_mix_var import Calculate_Moment
from alg.stroke_assemble.build_stroke_dataset import *
from alg.stroke_assemble.create_training_set import *
from alg.stroke_assemble.stroke_segmentation import *
from alg.stroke_assemble.assemble_mix import *
from alg.stroke_assemble.data_augmentation import Data_Augmentation 

import datetime
from threading import Thread
from time import sleep, ctime

from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)

zi2zi = Zi2ziFont( pic_path='',
                   device = device,
                   epoch = zi2zi_train_epoch,
                   base_zi2zi_path = os.path.join(alg_path,'./style_transfer/zi2zi'),
                   base_zi2zi_result_path = default_zi2zi_result_path )


text_inversion = TextualInversionTrain(trainData = '',
                                       base_dir = os.path.join(alg_path,'./style_transfer'),
                                       base_out_dir = default_ti_data_path,
                                       prompt="new_font",
                                       train_steps = sd_finetune_steps)

stage1a = Calculate_Moment()
stage1b = Build_Stokre_Dataset()
stage1c_a = Data_Augmentation()
stage1c_b = Create_Training_Set()
stage2 = Stroke_Segmentation()
stage3 = Stroke_Assemble(512,5)

@app.route('/progress',methods=['GET'])
def get_progress():
    progress = 0
    if current_step == 1:
        progress = zi2zi.getTrainState()
    elif current_step == 2:
        progress = zi2zi.getInferState()
    elif current_step == 3:
        progress = text_inversion.getTrainState()
    elif current_step == 4:
        progress = text_inversion.getInferState()    
    return {
        "code":0,
        "msg":"ok",
        "data":{
            "current_is_handling":current_is_handling,
            "current_step": current_step,
            "progress": '{:.2f}'.format(progress)
        }
    }

@app.route('/vector_progress',methods=['GET'])
def get_vector_progress():
    progress = 0
    if current_vector_step == 1:
        progress = stage1a.getprogress()
    elif current_vector_step == 2:
        progress = stage1b.getprogress()
    elif current_vector_step == 3:
        progress = stage1c_b.getprogress()
    elif current_vector_step == 4:
        progress = stage2.getprogress()  
    elif current_vector_step == 5:
        progress = stage3.getprogress()   
    return {
        "code":0,
        "msg":"ok",
        "data":{
            "current_is_handling":current_vector_is_handling,
            "current_step": current_vector_step,
            "progress": '{:.2f}'.format(progress)
        }
    }

@app.route('/upload', methods=['POST'])
def upload_file():
    # 检查是否上传了文件
    if 'file' not in request.files:
        return {
            "code":10000,
            "msg": 'failed',
        }

    # 获取上传的文件
    file = request.files['file']

    # 检查文件是否是zip压缩包
    if file.filename.endswith('.zip'):
        # 解压文件到指定目录
        folder_name = str(uuid.uuid4())
        # 创建新的文件夹
        destination = os.path.join(data_path,'zips/') + folder_name
        os.makedirs(destination)
        
        with zipfile.ZipFile(file,'r') as zip_ref:
            zip_ref.extractall(destination)
        
        for parent,dirnames,filenames in os.walk(destination):
            for filename in filenames:
                os.rename(os.path.join(parent,filename),os.path.join(parent,filename.encode('cp437').decode('gbk')))
            
        return {
            "code":0,
            "msg":"ok",
            "data":{
                "dir": 'zips/' + folder_name
            }
        }
    else:
        return {
            "code":10000,
            "msg": 'failed',
        }

@app.route('/start',methods=['POST'])
def start():
    global current_is_handling,current_step,current_vector_is_handling
    if current_is_handling or current_vector_is_handling:
        return {
        "code":10000,
        "msg":"current is handling",
    }
    current_is_handling = True
    
    path = os.path.join(data_path,request.json.get('path'))
    json_path = os.path.join(data_path,request.json.get('json_path'))
    if not os.path.exists(path) or not os.path.exists(json_path):
        return {
            "code":10001,
            "msg":"path not exists",
        }
        
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
    
    makedir(os.path.join(result_data_path,timestamp))
    makedir(os.path.join(result_data_path,f'{timestamp}/style_transfer'))
    makedir(os.path.join(result_data_path,f'{timestamp}/style_transfer/zi2zi_finetune'))
    makedir(os.path.join(result_data_path,f'{timestamp}/style_transfer/zi2zi_finetune/data'))
    makedir(os.path.join(result_data_path,f'{timestamp}/style_transfer/zi2zi_finetune/finetune'))
    makedir(os.path.join(result_data_path,f'{timestamp}/style_transfer/zi2zi_finetune/finetune/data'))
    makedir(os.path.join(result_data_path,f'{timestamp}/style_transfer/zi2zi_finetune/infer_data'))
    makedir(os.path.join(result_data_path,f'{timestamp}/style_transfer/zi2zi_finetune/train_result'))
    makedir(os.path.join(result_data_path,f'{timestamp}/style_transfer/sd_finetune'))
    makedir(os.path.join(result_data_path,f'{timestamp}/style_transfer/sd_finetune/infer_result'))
    
    zi2zi.set_result_path(os.path.join(result_data_path,f'{timestamp}/style_transfer/zi2zi_finetune'))
    text_inversion.setBaseOutDir(os.path.join(result_data_path,f'{timestamp}/style_transfer/sd_finetune'))
    
    zi2zi.zi2zi_main.train_process = 0
    zi2zi.zi2zi_main.infer_process = 0
    text_inversion.train_process = 0
    text_inversion.infer_process = 0
    
    moments_dir = os.path.join(result_data_path,f'{timestamp}/stroke_assemble/json_with_moments_var')
    stroke_img_dir = os.path.join(result_data_path,f'{timestamp}/stroke_assemble/stroke_img_var')
    stroke_dataset_dir = os.path.join(result_data_path,f'{timestamp}/stroke_assemble/json_stroke_dataset_var')
    segmentation_data_dir = os.path.join(result_data_path,f'{timestamp}/stroke_assemble/img_for_train')
    sd_img_dir = os.path.join(result_data_path,f'{timestamp}/style_transfer/sd_finetune/infer_result')
    segmentation_result_dir = os.path.join(result_data_path,f'{timestamp}/stroke_assemble/split_result')
    json_result_dir = os.path.join(result_data_path,f'{timestamp}/stroke_assemble/json_assemble')
    augmentation_dir = os.path.join(result_data_path,f'{timestamp}/stroke_assemble/json_augumentation')
    
    runs_log_dir = os.path.join(result_data_path,f'{timestamp}/stroke_assemble/runs_log')
    
    makedir(os.path.join(result_data_path,f'{timestamp}/stroke_assemble'))
    makedir(moments_dir)
    makedir(stroke_img_dir)
    makedir(stroke_dataset_dir)
    makedir(segmentation_data_dir)
    makedir(sd_img_dir)
    makedir(segmentation_result_dir)
    makedir(json_result_dir)
    makedir(augmentation_dir)
    
    stage1a.set_json_dir(json_path)
    stage1a.set_save_dir(moments_dir)
    stage1a.set_stroke_save_dir(stroke_img_dir)
    
    stage1b.set_moments_dir(moments_dir)
    stage1b.set_save_dir(stroke_dataset_dir)
    
    stage1c_a.set_dataset_path(stroke_dataset_dir)
    stage1c_a.set_save_dir(augmentation_dir)
    
    stage1c_b.set_moments_dir([moments_dir,augmentation_dir])
    stage1c_b.set_save_dir(segmentation_data_dir)
    
    stage2.set_train_data_dir(segmentation_data_dir)
    stage2.set_infer_data_dir(sd_img_dir)
    stage2.set_save_dir(segmentation_result_dir)
    stage2.set_device(device)
    stage2.set_runs_log_dir(runs_log_dir)
    
    stage3.set_split_result_dir(segmentation_result_dir)
    stage3.set_stroke_data_dir(stroke_dataset_dir)
    stage3.set_json_save_dir(json_result_dir)
      
    t1 = Thread(target=train_and_finetune_zi2zi, args=('train_and_finetune_zi2zi', path))
    t1.start()
    return {
        "code":0,
        "msg":"ok",
    }

@app.route('/result_dir',methods=['GET'])
def result_dir():
    return {
        "code":0,
        "msg":"ok",
        "data":{
            "zi2zi_dir": zi2zi.getInferResult(),
            "text_inversion_dir": text_inversion.getInferResult(),
            "charset": charset,
            "sd_infer_num": sd_infer_num,
            "stroke_json_dir": default_json_save_path if is_preview else stage3.json_save_dir,
        }
    }

@app.route('/file',methods=['GET'])
def get_file():
    if request.method == 'GET':
        relative_path = request.args.get("path")
        return send_file(relative_path,'image/png')

@app.route('/json_file',methods=['GET'])
def get_json_file():
    if request.method == 'GET':
        relative_path = request.args.get("path")
        return send_file(relative_path,mimetype='application/json')
    
def train_and_finetune_zi2zi(name,path):
    global zi2zi,current_step
    print('---开始---', name, '时间', ctime())
    current_step = 1
    zi2zi.set_picpath(path)
    zi2zi.main_train()
    print('***结束***', name, '时间', ctime())
    t2 = Thread(target=infer_zi2zi, args=('infer_zi2zi', path))
    t2.start()
    
def infer_zi2zi(name,path):
    global zi2zi,current_step
    print('---开始---', name, '时间', ctime())
    current_step = 2
    result = zi2zi.getCkptDir()
    zi2zi.infer(charset,zi2zi.ckpt_dir)
    print('***结束***', name, '时间', ctime())
    t3 = Thread(target=train_text_inversion, args=('train_text_inversion', path))
    t3.start()
    
def train_text_inversion(name,path):
    global text_inversion,current_step
    print('---开始---', name, '时间', ctime())
    current_step = 3
    text_inversion.setTrainData(path)
    text_inversion.train()
    print('***结束***', name, '时间', ctime())
    t4 = Thread(target=infer_text_inversion, args=('infer_text_inversion', path))
    t4.start()
    
def infer_text_inversion(name,path):
    global text_inversion,current_step,current_is_handling
    print('---开始---', name, '时间', ctime())
    current_step = 4
    model_dir = text_inversion.getCkptResult()
    image_dir = zi2zi.getInferResult()
    text_inversion.modelInfer(device,model_dir,image_dir,'qingxin',0.65,0.65,20,sd_infer_num)
    result = text_inversion.getCkptResult()
    current_is_handling = False
    current_step = 5
    t5 = Thread(target=stage1a_func, args=('stage1a', path))
    t5.start()
    print('***结束***', name, '时间', ctime())

def stage1a_func(name, timestamp):
    global stage1a,current_vector_step,current_vector_is_handling
    current_vector_is_handling = True
    print('---开始---', name, '时间', ctime())
    current_vector_step = 1
    stage1a.progress = 0
    save_dir, mean, std = stage1a.do()
    stage3.set_cnm_mean_std(mean,std)
    print('***结束***', name, '时间', ctime())
    t2 = Thread(target=stage1b_func, args=('stage1b',timestamp ))
    t2.start()

def stage1b_func(name,timestamp):
    global current_vector_step
    print('---开始---', name, '时间', ctime())
    current_vector_step = 2
    stage1b.do()
    print('***结束***', name, '时间', ctime())
    t3 = Thread(target=stage1c_func, args=('stage1c', timestamp ))
    t3.start()

def stage1c_func(name,timestamp):
    global current_vector_step
    print('---开始---', name, '时间', ctime())
    current_vector_step = 3
    stage1c_a.do()
    stage1c_b.do()
    print('***结束***', name, '时间', ctime())
    t4 = Thread(target=stage2_func, args=('stage2',timestamp ))
    t4.start()

def stage2_func(name,timestamp):
    global current_vector_step
    print('---开始---', name, '时间', ctime())
    current_vector_step = 4
    stage2.do(train_epochs=segmentation_train_epoch,load_epochs=segmentation_load_epoch)
    print('***结束***', name, '时间', ctime())
    t5 = Thread(target=stage3_func, args=('stage3', timestamp ))
    t5.start()

def stage3_func(name,timestamp):
    global current_vector_step,current_vector_is_handling
    print('---开始---', name, '时间', ctime())
    current_vector_step = 5 
    stage3.do()
    current_vector_is_handling = False
    current_vector_step = 6
    print('***结束***', name, '时间', ctime())
    
if __name__ == '__main__':
    app.run(port=8015,host="0.0.0.0")