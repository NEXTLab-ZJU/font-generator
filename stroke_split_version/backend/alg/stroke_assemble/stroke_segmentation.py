from torchvision import transforms
import torch.distributed as dist
from tensorboardX import SummaryWriter
from segmentation.data_loader.segmentation_dataset import SegmentationDataset
from segmentation.data_loader.transform import Rescale, ToTensor
from segmentation.trainer import Trainer
from segmentation.predict import *
from segmentation.models import all_models
from util.logger import Logger
import os
from tqdm import tqdm

class Stroke_Segmentation(object):
    def __init__(self):
        self.is_finished = False
        self.current_type = 0
        self.progress = 0
        
    def isfinished(self):
        return self.is_finished
    
    def set_device(self,device):
        self.device = device
    
    def getprogress(self):
        self.progress = (self.current_type*self.num_epochs+self.trainer.getprogress())/(self.num_type*self.num_epochs)
        return self.progress

    def set_train_data_dir(self,train_data_dir):
        self.train_data_dir = train_data_dir
    
    def set_infer_data_dir(self,infer_data_dir):
        self.infer_data_dir = infer_data_dir
        
    def set_save_dir(self,save_dir):
        self.save_dir = save_dir
    
    def set_runs_log_dir(self,runs_log_dir):
        self.runs_log_dir = runs_log_dir
    
    def do(self,train_data_dir = None,infer_data_dir = None,save_dir = None,train_epochs=20,load_epochs=20):
        if train_data_dir == None:
            train_data_dir = self.train_data_dir
        if infer_data_dir == None:
            infer_data_dir = self.infer_data_dir
        if save_dir == None:
            save_dir = self.save_dir
        
        model_name = 'fcn8_resnet34'
        image_axis_minimum_size=256
        batch_size = 4
        n_classes = 2
        num_epochs = train_epochs
        pretrained = False
        fixed_feature = False
        device = self.device

        compose = transforms.Compose([
                Rescale(image_axis_minimum_size),
                ToTensor()
            ])

        stroke_types = os.listdir(os.path.join(train_data_dir))
        stroke_types.remove('all')
        stroke_types.sort()

        self.num_epochs = num_epochs
        self.num_type = len(stroke_types)

        for progess,stroke_type in enumerate(stroke_types):
            self.current_type = progess

            train_images=os.path.join(train_data_dir,'all')
            train_labeled=os.path.join(train_data_dir,stroke_type)
            train_datasets = SegmentationDataset(train_images, train_labeled, n_classes, compose)
            train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

            logger = Logger(model_name='{}_{}'.format(stroke_type,model_name), data_name='example',runs=self.runs_log_dir)

            ### Model
            model = all_models.model_from_name[model_name](n_classes, batch_size,
                                                        pretrained=pretrained,
                                                        fixed_feature=fixed_feature)
            model.to(device)

            if pretrained and fixed_feature: #fine tunning
                params_to_update = model.parameters()
                print("Params to learn:")
                params_to_update = []
                for name, param in model.named_parameters():
                    if param.requires_grad == True:
                        params_to_update.append(param)
                        print("\t", name)
                optimizer = torch.optim.Adadelta(params_to_update)
            else:
                optimizer = torch.optim.Adadelta(model.parameters())
                
            self.trainer = Trainer(model, optimizer, logger, num_epochs, train_loader, None,'txt_log')
            self.trainer.train()

            logger.load_model(model, 'epoch_'+str(load_epochs-1))
        
            for img_path in tqdm(os.listdir(infer_data_dir)):
                predict(model, os.path.join(infer_data_dir,img_path),
                        os.path.join(save_dir,stroke_type,img_path))

        self.is_finished = True
        self.progress = 1
        return save_dir