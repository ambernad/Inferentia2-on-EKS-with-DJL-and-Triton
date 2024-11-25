import torch
import tempfile
import os
import time
import copy
import boto3
import sys
import numpy as np
from djl_python import Input, Output
import torch.nn as nn
import torch_neuronx
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.models.attention_processor import Attention
 
class UNetWrap(nn.Module):
    def __init__(self, unet):
        super().__init__()
        self.unet = unet

    def forward(self, sample, timestep, encoder_hidden_states, text_embeds=None, time_ids=None):
        out_tuple = self.unet(sample,
                              timestep,
                              encoder_hidden_states,
                              added_cond_kwargs={"text_embeds": text_embeds, "time_ids": time_ids},
                              return_dict=False)
        return out_tuple
    
    
class NeuronUNet(nn.Module):
    def __init__(self, unetwrap):
        super().__init__()
        self.unetwrap = unetwrap
        self.config = unetwrap.unet.config
        self.in_channels = unetwrap.unet.in_channels
        self.add_embedding = unetwrap.unet.add_embedding
        self.device = unetwrap.unet.device

    def forward(self, sample, timestep, encoder_hidden_states, added_cond_kwargs=None, return_dict=False, cross_attention_kwargs=None):
        sample = self.unetwrap(sample,
                               timestep.float().expand((sample.shape[0],)),
                               encoder_hidden_states,
                               added_cond_kwargs["text_embeds"],
                               added_cond_kwargs["time_ids"])[0]
        return UNet2DConditionOutput(sample=sample)

pipe = None

def load_model(properties):
    model_id = "stabilityai/stable-diffusion-xl-base-1.0"
    decoder_filename = "vae_decoder.pt"
    unet_filename = "unet.pt"
    post_quant_conv_filename = "vae_post_quant_conv.pt"

    pipe = DiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.unet = NeuronUNet(UNetWrap(pipe.unet))
    device_ids = [0,1]
    pipe.unet.unetwrap = torch.jit.load(unet_filename)
    pipe.unet.unetwrap = torch_neuronx.DataParallel(pipe.unet.unetwrap, device_ids, set_dynamic_batching=False)
    return pipe

def infer(prompt):
    total_time = 0
    start_time = time.time()
    image = pipe(prompt).images[0]
    total_time = time.time() - start_time

    # Generate image key with current timestamp up to milliseconds
    current_time = time.strftime("%Y%m%d%H%M%S%f")[:-3]
    image_key = f"generated_image_{current_time}.png"
    bucket_name = "abernads-sd2-images"

    # Save image to local file
    image_path = "image.png"
    image.save(image_path)

    # Upload image to S3 bucket
    s3 = boto3.client('s3')
    s3.upload_file(image_path, bucket_name, image_key)
    
    result = "Inference time: " + str(np.round(total_time, 2)) + ". Image saved to S3 bucket " + bucket_name + " with key " + image_key 
    return {'result': result}

def handle(inputs: Input):
    global pipe
    if not pipe:
        pipe = load_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None

    data = inputs.get_as_json()
    prompt = data.pop("prompt")
    print(prompt)
    outputs = infer(prompt)
    result = {"outputs": outputs}
    return Output().add_as_json(result)