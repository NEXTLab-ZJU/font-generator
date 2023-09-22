from .data.dataset import DatasetFromObj
from torch.utils.data import DataLoader, TensorDataset
from .model import Zi2ZiModel
import os
import sys
import argparse
import torch
import random
import time
import math
import os
import torchvision.utils as vutils

def chkormakedir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class zi2ziMain(object):
    def __init__(self,epoch,device):
        parser = argparse.ArgumentParser(description='Train')
        parser.add_argument('--gpu_ids', default=["cuda:0"], nargs='+', help="GPUs")
        parser.add_argument('--image_size', type=int, default=256,
                            help="size of your input and output image")
        parser.add_argument('--L1_penalty', type=int, default=100, help='weight for L1 loss')
        parser.add_argument('--Lconst_penalty', type=int, default=15, help='weight for const loss')
        # parser.add_argument('--Ltv_penalty', dest='Ltv_penalty', type=float, default=0.0, help='weight for tv loss')
        parser.add_argument('--Lcategory_penalty', type=float, default=1.0,
                            help='weight for category loss')
        parser.add_argument('--embedding_num', type=int, default=40,
                            help="number for distinct embeddings")
        parser.add_argument('--embedding_dim', type=int, default=128, help="dimension for embedding")
        parser.add_argument('--batch_size', type=int, default=64, help='number of examples in batch')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for adam')
        parser.add_argument('--schedule', type=int, default=20, help='number of epochs to half learning rate')
        parser.add_argument('--freeze_encoder', action='store_true',
                            help="freeze encoder weights during training")
        parser.add_argument('--fine_tune', type=str, default=None,
                            help='specific labels id to be fine tuned')
        parser.add_argument('--inst_norm', action='store_true',
                            help='use conditional instance normalization in your model')
        parser.add_argument('--flip_labels', action='store_true',
                            help='whether flip training data labels or not, in fine tuning')
        parser.add_argument('--random_seed', type=int, default=777,
                            help='random seed for random and pytorch')
        parser.add_argument('--resume', type=int, default=None, help='resume from previous training')
        parser.add_argument('--input_nc', type=int, default=1,
                            help='number of input images channels')
        self.parser = parser
        self.train_process = 0
        self.infer_process = 0
        #模型保存目录
        #self.output_dir = savePath
        #模型训练数据地址
        #self.train_data_dir = dataDir
        #模型训练轮次
        self.epoch = epoch
        self.device = device
        #模型保存地址
        '''
        self.ckpt_dir = os.path.join(self.output_dir, "checkpoint")
        self.infer_dir = infer_dir
        if not os.path.exists(infer_dir):
            os.makedirs(infer_dir)
        chkormakedir(self.ckpt_dir)
        '''
    def getTrainState(self):
        return self.train_process
    def getInferState(self):
        return self.infer_process
    '''
    图像处理列表
    类别列表
    infer保存目录
    src字符数组
    ckpt_dir模型地址
    '''
    def infer(self,img_list,label_list,ckpt_dir,infer_dir,src,epoch):
        args = self.parser.parse_args()
        model = Zi2ZiModel(
            input_nc=args.input_nc,
            embedding_num=args.embedding_num,
            embedding_dim=args.embedding_dim,
            Lconst_penalty=args.Lconst_penalty,
            Lcategory_penalty=args.Lcategory_penalty,
            save_dir=ckpt_dir,
            gpu_ids=[self.device],
            is_training=False)
        model.setup()
        #model.print_networks(True)
        model.load_networks(epoch)
        dataset = TensorDataset(label_list, img_list, img_list)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        for bindex,batch in enumerate(dataloader):
            cnt = bindex*args.batch_size
            with torch.no_grad():
                model.set_input(batch[0], batch[2], batch[1])
                model.forward()
                tensor_to_plot = model.fake_B
                for label, image_tensor in zip(batch[0], tensor_to_plot):
                    vutils.save_image(image_tensor, os.path.join(infer_dir, str(src[cnt]) + '.png'))
                    cnt += 1
                    self.infer_process = cnt/int(img_list.shape[0])
                    print(self.infer_process)
        self.infer_process = 1
        print(self.infer_process)
    
    '''
    resume 0 训练 1 fintune
    '''
    def train(self,learn_rate,checkpoint_dir,data_dir,resume=0):
        args = self.parser.parse_args()
        random.seed(args.random_seed)
        torch.manual_seed(args.random_seed)
        start_time = time.time()
        model = Zi2ZiModel(
            input_nc=args.input_nc,
            embedding_num=args.embedding_num,
            embedding_dim=args.embedding_dim,
            Lconst_penalty=args.Lconst_penalty,
            Lcategory_penalty=args.Lcategory_penalty,
            save_dir=checkpoint_dir,
            lr=learn_rate,
            gpu_ids=[self.device]
        )
        model.setup()
        #model.print_networks(True)
        if resume:
            model.load_networks(self.epoch)

        # val dataset load only once, no shuffle
        val_dataset = DatasetFromObj(os.path.join(data_dir, 'val.obj'), input_nc=args.input_nc)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
        for epoch in range(self.epoch):
            # generate train dataset every epoch so that different styles of saved char imgs can be trained.
            train_dataset = DatasetFromObj(
                os.path.join(data_dir, 'train.obj'),
                input_nc=args.input_nc,
                augment=True,
                bold=False,
                rotate=False,
                blur=True,
            )
            total_batches = math.ceil(len(train_dataset) / args.batch_size)
            dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
            for bid, batch in enumerate(dataloader):
                model.set_input(batch[0], batch[2], batch[1])
                const_loss, l1_loss, category_loss, cheat_loss = model.optimize_parameters()
                if self.epoch==1:
                    break
                if bid % 100 == 0:
                    passed = time.time() - start_time
                    log_format = "Epoch: [%2d], [%4d/%4d] time: %4.2f, d_loss: %.5f, g_loss: %.5f, " + \
                                "category_loss: %.5f, cheat_loss: %.5f, const_loss: %.5f, l1_loss: %.5f"
                    print(log_format % (epoch, bid, total_batches, passed, model.d_loss.item(), model.g_loss.item(),
                                        category_loss, cheat_loss, const_loss, l1_loss))
                self.train_process = ((epoch * total_batches + bid) / ((self.epoch) * total_batches))

            if self.epoch==1:
                break    
            if (epoch + 1) % args.schedule == 0:
                model.update_lr()
        model.save_networks(self.epoch)
        self.train_process = 1
        print(self.train_process)
            
