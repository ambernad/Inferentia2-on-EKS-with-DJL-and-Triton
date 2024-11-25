import triton_python_backend_utils as pb_utils
import torch
import json
import tempfile
import os
import time
import copy
import boto3
import sys
import numpy as np
import torch.nn as nn
import torch_neuronx
from optimum.neuron import NeuronStableDiffusionXLPipeline

os.environ["NEURON_RT_NUM_CORES"] = "1"

class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.

        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device
            ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        
        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                json.loads(args["model_config"]), "result"
            )["data_type"]
        )

        model_id = "/models/sdxl/1/sdxl_neuron"
        input_shapes = {"batch_size": 1, "height": 1024, "width": 1024}
        print ("device pick initiated")
        device_ids = [0,1,2,3]
        self.stable_diffusion = NeuronStableDiffusionXLPipeline.from_pretrained(model_id, export=False, **input_shapes, device_ids=device_ids)
        #self.stable_diffusion = NeuronStableDiffusionXLPipeline.from_pretrained(model_id, export=False, **input_shapes)
        print("load complete")

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model.

        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest

        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """
        responses = []
        for request in requests:
            inp = pb_utils.get_input_tensor_by_name(request, "prompt")
            prompt = inp.as_numpy()[0][0].decode()
        total_time = 0
        start_time = time.time()
        print("start inference")
        image = self.stable_diffusion(prompt).images[0]
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
        response = pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("result",np.array(result, dtype=self.output_dtype),)])
        responses.append(response)
        return responses