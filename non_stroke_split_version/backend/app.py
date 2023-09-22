from flask import Flask, request,send_file
import os
import zipfile
import uuid
from config import *
from util import *
import sys 
sys.path.append('.')
sys.path.append(os.path.join(alg_path,'./style_transfer'))

from alg.style_transfer.zi2zi_fintune import *
from alg.style_transfer.text_inversion_fintune import *

import datetime
from threading import Thread
from time import sleep, ctime

from flask_cors import CORS, cross_origin
app = Flask(__name__)
cors = CORS(app)

zi2zi = Zi2ziFont( pic_path='',
                   device=device,
                   epoch = zi2zi_train_epoch,
                   base_zi2zi_path = os.path.join(alg_path,'./style_transfer/zi2zi'),
                   base_zi2zi_result_path = default_zi2zi_result_path )


text_inversion = TextualInversionTrain(trainData = '',
                                       base_dir = os.path.join(alg_path,'./style_transfer'),
                                       base_out_dir = default_ti_data_path,
                                       prompt="new_font",
                                       train_steps = sd_finetune_steps)

    
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
    global current_is_handling,zi2zi,current_step
    if current_is_handling:
        return {
            "code":10000,
            "msg":"current is handling",
        }
    current_is_handling = True
    
    path = os.path.join(data_path,request.json.get('path'))
    if not os.path.exists(path):
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
            "sd_infer_num": sd_infer_num
        }
    }

@app.route('/file',methods=['GET'])
def get_zi2zi_file():
    if request.method == 'GET':
        relative_path = request.args.get("path")
        return send_file(relative_path,'image/png')
    
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
    print('***结束***', name, '时间', ctime())


if __name__ == '__main__':
    app.run(port=8015,host="127.0.0.1")