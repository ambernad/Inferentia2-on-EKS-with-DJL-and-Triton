import triton_python_backend_utils as pb_utils
import time
import torch
import json
import numpy as np
from transformers import AutoTokenizer
from transformers_neuronx.llama.model import LlamaForSampling



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

        model_id = "/models/llama2-13b/1/llama2-13b-split"
        # load meta-llama/Llama-2-13b to the NeuronCores with 24-way tensor parallelism and run compilation
        self.neuron_model = LlamaForSampling.from_pretrained(model_id, batch_size=1, context_length_estimate=32, tp_degree=12, amp='f16')
        self.neuron_model.to_neuron()

        # construct a tokenizer and encode prompt text
        self.tokenizer = AutoTokenizer.from_pretrained('/models/llama2-13b/1/llama2-13b-orig')
        prompt = "Hello, I'm a language model,"
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # run inference with top-k sampling
        with torch.inference_mode():
            start = time.time()
            generated_sequences = self.neuron_model.sample(input_ids, sequence_length=2048, top_k=50)
            elapsed = time.time() - start

        generated_sequences = [self.tokenizer.decode(seq) for seq in generated_sequences]
        #generated_sequences = [seq.encode('utf-8').decode('utf-8') for seq in generated_sequences]
        #print(f'generated sequences {generated_sequences} in {elapsed} seconds')
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
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        # run inference with top-k sampling
        with torch.inference_mode():
            start = time.time()
            generated_sequences = self.neuron_model.sample(input_ids, sequence_length=2048, top_k=50)
            elapsed = time.time() - start

        generated_sequences = [self.tokenizer.decode(seq) for seq in generated_sequences]
        generated_sequences = [seq.encode('utf-8').decode('utf-8') for seq in generated_sequences]
        #print(f'generated sequences {generated_sequences} in {elapsed} seconds')
        response = pb_utils.InferenceResponse(output_tensors=[pb_utils.Tensor("result",np.array(generated_sequences, dtype=self.output_dtype),)])
        responses.append(response)
        return responses