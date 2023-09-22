#!/usr/bin/env python
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import PIL
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from huggingface_hub import HfFolder, Repository, create_repo, whoami

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from multiprocessing import Process
import time
import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from diffusers import StableDiffusionImg2ImgPipeline                                                                                                                                                                                                                     

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.13.0.dev0")

logger = get_logger(__name__)


def save_progress(text_encoder, placeholder_token_id, accelerator, placeholder_token, save_path):
    logger.info("Saving embeddings")
    learned_embeds = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[placeholder_token_id]
    learned_embeds_dict = {placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)


    



def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=3000,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--only_save_embeds",
        action="store_true",
        default=False,
        help="Save only the embeddings for the new concept.",
    )
    
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    
    
    
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    
    parser.add_argument("--num_train_epochs", type=int, default=100)
    
    
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=50,
        help=(
            "Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    

    return args


imagenet_templates_small = [
    "a photo of a {}",
    "a rendering of a {}",
    "a cropped photo of the {}",
    "the photo of a {}",
    "a photo of a clean {}",
    "a photo of a dirty {}",
    "a dark photo of the {}",
    "a photo of my {}",
    "a photo of the cool {}",
    "a close-up photo of a {}",
    "a bright photo of the {}",
    "a cropped photo of a {}",
    "a photo of the {}",
    "a good photo of the {}",
    "a photo of one {}",
    "a close-up photo of the {}",
    "a rendition of the {}",
    "a photo of the clean {}",
    "a rendition of a {}",
    "a photo of a nice {}",
    "a good photo of a {}",
    "a photo of the nice {}",
    "a photo of the small {}",
    "a photo of the weird {}",
    "a photo of the large {}",
    "a photo of a cool {}",
    "a photo of a small {}",
]

imagenet_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]


class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.templates = imagenet_style_templates_small if learnable_property == "style" else imagenet_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"




class TextualInversionTrain(object):
    '''
    初始化训练相关参数
    '''
    def __init__(self,trainData,base_dir='./',train_steps=3000,base_out_dir='./hanzi_result/',prompt="qingxin"):
        self.placeholder_token = prompt
        #预训练模型
        self.pretrained_model_name_or_path = os.path.join(base_dir,"stablediffusion/base_model")
        self.train_data_dir = trainData
        #Ti模型保存位置
        self.base_output_dir = os.path.join(base_out_dir,'model')
        self.output_dir = os.path.join(self.base_output_dir,"zd_"+prompt)
        self.logging_dir = os.path.join(base_out_dir,'logging')
        self.learnable_property = "object"
        self.initializer_token = "font"
        self.resolution = 512
        self.train_batch_size = 1
        self.gradient_accumulation_steps =1
        self.max_train_steps = train_steps
        self.learning_rate = 5e-06
        #self.lr_scheduler = "constant"
        self.lr_warmup_steps = 0
        #训练进度
        self.train_process = 0
        #测试图片保存路径
        self.infer_result = os.path.join(base_out_dir,'infer_result')
        #预测进度
        self.infer_process = 0
        
        if self.train_data_dir is None:
            raise ValueError("You must specify a train data directory.")
    
    def setBaseOutDir(self,base_out_dir='./hanzi_result/'):
        self.base_out_dir = base_out_dir
        self.base_output_dir = os.path.join(base_out_dir,'model')
        self.output_dir = os.path.join(self.base_output_dir,"zd_"+ self.placeholder_token)
        self.infer_result = os.path.join(base_out_dir,'infer_result')
        self.logging_dir = os.path.join(base_out_dir,'logging')
        if not os.path.exists(self.logging_dir):
            os.makedirs(self.logging_dir)
            
        if not os.path.exists(self.base_output_dir):
            os.makedirs(self.base_output_dir)
        
        if not os.path.exists(self.infer_result):
            os.makedirs(self.infer_result)

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    '''
    设置训练数据目录
    '''
    def setTrainData(self,train_data_dir):
        self.train_data_dir = train_data_dir

    '''
    获取inferresult
    ''' 
    def getInferResult(self):
        return self.infer_result
    '''
    获取训练的ckpt路径
    '''
    def getCkptResult(self):
        return self.output_dir
    '''
    获取训练的百分比
    '''
    def getTrainState(self):
        print(self.train_process)
        return self.train_process
    
    '''
    获取Ti结果图位置
    '''
    def getInferResult(self):
        return self.infer_result
    
    '''
    获取Ti全部模型位置
    '''
    def getAllCkpts(self):
        ckptlist = os.listdir(self.base_output_dir)
        return [self.base_output_dir+item for item in ckptlist]
    
    '''
    获取相应的prompt
    '''
    def getPrompt(self):
        ckpt_list = os.listdir(self.base_output_dir)
        return [ckpt_list.split('_')[-1] for item in ckpt_list]

    '''
    获取infer的百分比
    '''
    def getInferState(self):
        return self.infer_process
    '''
    Ti图像预测生成
    '''
    def modelInfer(self,device,modelDir,imgDir,prompt,guidance_scale,strength,num_inference_steps,batch_size=64):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(modelDir, torch_dtype=torch.float16).to(
         device
        )
        if prompt == None or prompt == '':
            return "ERROR:PROMPT NONE"
        
        if guidance_scale == None or guidance_scale == '':
            guidance_scale = 7
        else:
            guidance_scale = float(guidance_scale)

        if strength == None or strength == '':
            strength = 0.65
        else:
            strength = float(strength)

        if num_inference_steps == None or num_inference_steps == '':
            num_inference_steps = 36
        else:
            num_inference_steps = int(num_inference_steps)

        if batch_size == None or batch_size == '':
            batch_size = 1
        else:
            batch_size = int(batch_size)
            
        piclist = []
        pitemlist = []
        for pitem in os.listdir(imgDir):
            pitemlist.append(pitem)
            piclist.append(os.path.join(imgDir,pitem))     
        
        round_inter = len(piclist)/batch_size if len(piclist)%batch_size==0 else len(piclist)//batch_size+1
        round_inter = int(round_inter)
        for iter_num in range(round_inter):
            batch_pic = []
            infer_list = []
            if iter_num == round_inter-1:
                infer_list = piclist[batch_size*iter_num:]
            else:
                infer_list = piclist[batch_size*iter_num:batch_size*(iter_num+1)]

            for in_item in infer_list:
                init_image = Image.open(in_item).convert('RGB')
                init_image = init_image.resize((512,512),Image.ANTIALIAS)
                init_image.thumbnail((768, 768))
                batch_pic.append(init_image)

            prompts = batch_size * [prompt]
            print(batch_size)
            print(prompts)
            imgs = pipe(
                prompt=prompts,
                image=batch_pic, 
                strength=strength, 
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps).images
            print(len(imgs))
            for i,item in enumerate(imgs):
                print("self.infer_result+'/'+ pitemlist[batch_size*iter_num+i]",self.infer_result+'/'+ pitemlist[batch_size*iter_num+i])
                imgs[i].save(self.infer_result+'/'+ pitemlist[batch_size*iter_num+i].split('.')[0] + '_' + str(i) + '.png')
            
            self.infer_process  = iter_num/round_inter

        self.infer_process  = 1

    def train(self):
        args = parse_args()
        logging_dir = os.path.join(self.output_dir, self.logging_dir)

        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            mixed_precision=args.mixed_precision,
            log_with=args.report_to,
            project_dir=logging_dir,
        )

        if args.report_to == "wandb":
            if not is_wandb_available():
                raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
            import wandb

        # Make one log on every process with the configuration for debugging.
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        # If passed along, set the training seed now.
        if args.seed is not None:
            set_seed(args.seed)

        # Handle the repository creation
        if accelerator.is_main_process:
            if args.push_to_hub:
                if args.hub_model_id is None:
                    repo_name = get_full_repo_name(Path(self.output_dir).name, token=args.hub_token)
                else:
                    repo_name = args.hub_model_id
                create_repo(repo_name, exist_ok=True, token=args.hub_token)
                repo = Repository(self.output_dir, clone_from=repo_name, token=args.hub_token)

                with open(os.path.join(self.output_dir, ".gitignore"), "w+") as gitignore:
                    if "step_*" not in gitignore:
                        gitignore.write("step_*\n")
                    if "epoch_*" not in gitignore:
                        gitignore.write("epoch_*\n")
            elif self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)

        # Load tokenizer
        if args.tokenizer_name:
            tokenizer = CLIPTokenizer.from_pretrained(args.tokenizer_name)
        elif self.pretrained_model_name_or_path:
            tokenizer = CLIPTokenizer.from_pretrained(self.pretrained_model_name_or_path, subfolder="tokenizer")

        # Load scheduler and models
        noise_scheduler = DDPMScheduler.from_pretrained(self.pretrained_model_name_or_path, subfolder="scheduler")
        text_encoder = CLIPTextModel.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision
        )
        vae = AutoencoderKL.from_pretrained(self.pretrained_model_name_or_path, subfolder="vae", revision=args.revision)
        unet = UNet2DConditionModel.from_pretrained(
            self.pretrained_model_name_or_path, subfolder="unet", revision=args.revision
        )

        # Add the placeholder token in tokenizer
        num_added_tokens = tokenizer.add_tokens(self.placeholder_token)
        if num_added_tokens == 0:
            raise ValueError(
                f"The tokenizer already contains the token {self.placeholder_token}. Please pass a different"
                " `placeholder_token` that is not already in the tokenizer."
            )

        # Convert the initializer_token, placeholder_token to ids
        token_ids = tokenizer.encode(self.initializer_token, add_special_tokens=False)
        # Check if initializer_token is a single token or a sequence of tokens
        if len(token_ids) > 1:
            raise ValueError("The initializer token must be a single token.")

        initializer_token_id = token_ids[0]
        placeholder_token_id = tokenizer.convert_tokens_to_ids(self.placeholder_token)

        # Resize the token embeddings as we are adding new special tokens to the tokenizer
        text_encoder.resize_token_embeddings(len(tokenizer))

        # Initialise the newly added placeholder token with the embeddings of the initializer token
        token_embeds = text_encoder.get_input_embeddings().weight.data
        token_embeds[placeholder_token_id] = token_embeds[initializer_token_id]

        # Freeze vae and unet
        vae.requires_grad_(False)
        unet.requires_grad_(False)
        # Freeze all parameters except for the token embeddings in text encoder
        text_encoder.text_model.encoder.requires_grad_(False)
        text_encoder.text_model.final_layer_norm.requires_grad_(False)
        text_encoder.text_model.embeddings.position_embedding.requires_grad_(False)

        if args.gradient_checkpointing:
            # Keep unet in train mode if we are using gradient checkpointing to save memory.
            # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
            unet.train()
            text_encoder.gradient_checkpointing_enable()
            unet.enable_gradient_checkpointing()

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                unet.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available. Make sure it is installed correctly")

        # Enable TF32 for faster training on Ampere GPUs,
        # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
        if args.allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True

        if args.scale_lr:
            self.learning_rate = (
                self.learning_rate * self.gradient_accumulation_steps * self.train_batch_size * accelerator.num_processes
            )

        # Initialize the optimizer
        optimizer = torch.optim.AdamW(
            text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
            lr=self.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        # Dataset and DataLoaders creation:
        train_dataset = TextualInversionDataset(
            data_root=self.train_data_dir,
            tokenizer=tokenizer,
            size=self.resolution,
            placeholder_token=self.placeholder_token,
            repeats=args.repeats,
            learnable_property=self.learnable_property,
            center_crop=args.center_crop,
            set="train",
        )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
        )

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.gradient_accumulation_steps)
        if self.max_train_steps is None:
            self.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=self.lr_warmup_steps * self.gradient_accumulation_steps,
            num_training_steps=self.max_train_steps * self.gradient_accumulation_steps,
        )

        # Prepare everything with our `accelerator`.
        text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            text_encoder, optimizer, train_dataloader, lr_scheduler
        )

        # For mixed precision training we cast the unet and vae weights to half-precision
        # as these models are only used for inference, keeping weights in full precision is not required.
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        # Move vae and unet to device and cast to weight_dtype
        unet.to(accelerator.device, dtype=weight_dtype)
        vae.to(accelerator.device, dtype=weight_dtype)

        # We need to recalculate our total training steps as the size of the training dataloader may have changed.
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.gradient_accumulation_steps)
        if overrode_max_train_steps:
            self.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        # Afterwards we recalculate our number of training epochs
        args.num_train_epochs = math.ceil(self.max_train_steps / num_update_steps_per_epoch)

        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            accelerator.init_trackers("textual_inversion", config=vars(args))

        # Train!
        total_batch_size = self.train_batch_size * accelerator.num_processes * self.gradient_accumulation_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {self.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {self.max_train_steps}")
        global_step = 0
        first_epoch = 0

        # Potentially load in the weights and states from a previous save
        if args.resume_from_checkpoint:
            if args.resume_from_checkpoint != "latest":
                path = os.path.basename(args.resume_from_checkpoint)
            else:
                # Get the most recent checkpoint
                dirs = os.listdir(self.output_dir)
                dirs = [d for d in dirs if d.startswith("checkpoint")]
                dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
                path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                accelerator.print(
                    f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
                )
                args.resume_from_checkpoint = None
            else:
                accelerator.print(f"Resuming from checkpoint {path}")
                accelerator.load_state(os.path.join(self.output_dir, path))
                global_step = int(path.split("-")[1])

                resume_global_step = global_step * self.gradient_accumulation_steps
                first_epoch = global_step // num_update_steps_per_epoch
                resume_step = resume_global_step % (num_update_steps_per_epoch * self.gradient_accumulation_steps)

        # Only show the progress bar once on each machine.
        #progress_bar = tqdm(range(global_step, self.max_train_steps), disable=not accelerator.is_local_main_process)
        #progress_bar.set_description("Steps")

        # keep original embeddings as reference
        orig_embeds_params = accelerator.unwrap_model(text_encoder).get_input_embeddings().weight.data.clone()

        for epoch in range(first_epoch, args.num_train_epochs):
            text_encoder.train()
            for step, batch in enumerate(train_dataloader):
                self.train_process = step / (self.max_train_steps)
                print(self.train_process)
                # Skip steps until we reach the resumed step
                if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                    if step % self.gradient_accumulation_steps == 0:
                        #progress_bar.update(1)
                        pass
                    continue

                with accelerator.accumulate(text_encoder):
                    # Convert images to latent space
                    latents = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist.sample().detach()
                    latents = latents * vae.config.scaling_factor

                    # Sample noise that we'll add to the latents
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each image
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Get the text embedding for conditioning
                    encoder_hidden_states = text_encoder(batch["input_ids"])[0].to(dtype=weight_dtype)

                    # Predict the noise residual
                    model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                    # Get the target for loss depending on the prediction type
                    if noise_scheduler.config.prediction_type == "epsilon":
                        target = noise
                    elif noise_scheduler.config.prediction_type == "v_prediction":
                        target = noise_scheduler.get_velocity(latents, noise, timesteps)
                    else:
                        raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    accelerator.backward(loss)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                    # Let's make sure we don't update any embedding weights besides the newly added token
                    index_no_updates = torch.arange(len(tokenizer)) != placeholder_token_id
                    with torch.no_grad():
                        accelerator.unwrap_model(text_encoder).get_input_embeddings().weight[
                            index_no_updates
                        ] = orig_embeds_params[index_no_updates]

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    #progress_bar.update(1)
                    global_step += 1
                    if global_step % args.save_steps == 0:
                        save_path = os.path.join(self.output_dir, f"learned_embeds-steps-{global_step}.bin")
                        save_progress(text_encoder, placeholder_token_id, accelerator, self.placeholder_token, save_path)

                    if global_step % args.checkpointing_steps == 0:
                        if accelerator.is_main_process:
                            save_path = os.path.join(self.output_dir, f"checkpoint-{global_step}")
                            accelerator.save_state(save_path)
                            logger.info(f"Saved state to {save_path}")

                logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
                #progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

                if global_step >= self.max_train_steps:
                    break

            if args.validation_prompt is not None and epoch % args.validation_epochs == 0:
                logger.info(
                    f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
                    f" {args.validation_prompt}."
                )
                # create pipeline (note: unet and vae are loaded again in float32)
                pipeline = DiffusionPipeline.from_pretrained(
                    self.pretrained_model_name_or_path,
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    tokenizer=tokenizer,
                    unet=unet,
                    vae=vae,
                    revision=args.revision,
                    torch_dtype=weight_dtype,
                )
                pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
                pipeline = pipeline.to(accelerator.device)
                #pipeline.set_progress_bar_config(disable=True)

                # run inference
                generator = (
                    None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
                )
                images = []
                for _ in range(args.num_validation_images):
                    with torch.autocast("cuda"):
                        image = pipeline(args.validation_prompt, num_inference_steps=25, generator=generator).images[0]
                    images.append(image)

                for tracker in accelerator.trackers:
                    if tracker.name == "tensorboard":
                        np_images = np.stack([np.asarray(img) for img in images])
                        tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
                    if tracker.name == "wandb":
                        tracker.log(
                            {
                                "validation": [
                                    wandb.Image(image, caption=f"{i}: {args.validation_prompt}")
                                    for i, image in enumerate(images)
                                ]
                            }
                        )

                del pipeline
                torch.cuda.empty_cache()

        self.train_process = 1
        
        # Create the pipeline using using the trained modules and save it.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            if args.push_to_hub and args.only_save_embeds:
                logger.warn("Enabling full model saving because --push_to_hub=True was specified.")
                save_full_model = True
            else:
                save_full_model = not args.only_save_embeds
            if save_full_model:
                pipeline = StableDiffusionPipeline.from_pretrained(
                    self.pretrained_model_name_or_path,
                    text_encoder=accelerator.unwrap_model(text_encoder),
                    vae=vae,
                    unet=unet,
                    tokenizer=tokenizer,
                )
                pipeline.save_pretrained(self.output_dir)
            # Save the newly trained embeddings
            save_path = os.path.join(self.output_dir, "learned_embeds.bin")
            save_progress(text_encoder, placeholder_token_id, accelerator, self.placeholder_token, save_path)

            if args.push_to_hub:
                repo.push_to_hub(commit_message="End of training", blocking=False, auto_lfs_prune=True)
        
        accelerator.end_training()

if __name__ == "__main__":
    #初始化
    
    #参数是训练数据dir
    '''
    test = TextualInversionTrain('../../apple_offline_backend/data/zips/8010e61f-a392-4836-87d2-a2678be5b68f',base_dir="./")

    #训练
    test.train()
    '''
    '''
    #获取训练进度
    result = test.getTrainState()

    #获取ckpt位置
    result = test.getCkptResult()

    #预测
    device cuda地址
    result 即diffuse模型地址
    imgDir 即zi2zi的生成底图位置
    其他参数按之前的
    test.modelInfer(device,result,'./alg/hanzi_train/zi2zi_data/infer_data','qingxin',0.65,0.65,20,1)

    #获取预测进度
    test.getInferState()

    #获取预测结果位置
    test.getInferResult()
    '''
    '''
    device = "cuda:1"

    path = "/opt/data/private/apple_offline_backend/data/zips/8010e61f-a392-4836-87d2-a2678be5b68f"

    test = TextualInversionTrain(path,base_dir="./alg/hanzi_train",train_steps=10,base_out_dir='./alg/hanzi_result/')
    
    #train_steps =10 有默认值，测试可以改小，实际用时不用设置

    test.train()
    result = test.getTrainState()
    print(result)
    result = test.getCkptResult()
    print(result)

    test.modelInfer(device,result,'./alg/hanzi_train/zi2zi_data/infer_data','qingxin',0.65,0.65,20,1)
    result = test.getInferState()
    print(result)
    result = test.getInferResult()
    print(result)

        '''

    

    


    
