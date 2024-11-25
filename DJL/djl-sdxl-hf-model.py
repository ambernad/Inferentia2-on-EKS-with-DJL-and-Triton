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
from optimum.neuron import NeuronStableDiffusionXLPipeline

stable_diffusion = None

def load_model(properties):
    model_id = "./sdxl_neuron"
    input_shapes = {"batch_size": 1, "height": 1024, "width": 1024}

    device_ids = [0,1,2,3]
    stable_diffusion = NeuronStableDiffusionXLPipeline.from_pretrained(model_id, export=False, **input_shapes, device_ids=device_ids)
    print("load complete")
    return stable_diffusion

def infer(prompt):
    total_time = 0
    start_time = time.time()
    print("start inference")
    image = stable_diffusion(prompt).images[0]
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
    global stable_diffusion
    if not stable_diffusion:
        stable_diffusion = load_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None

    data = inputs.get_as_json()
    prompt = data.pop("prompt")
    outputs = infer(prompt)
    result = {"outputs": outputs}
    return Output().add_as_json(result)