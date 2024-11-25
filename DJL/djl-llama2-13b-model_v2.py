import time
import torch
from transformers import AutoTokenizer
from djl_python import Input, Output
from transformers_neuronx.llama.model import LlamaForSampling

neuron_model = None

def load_model(properties):
    neuron_model = LlamaForSampling.from_pretrained('./Llama-2-13b-split', batch_size=1,context_length_estimate=32, tp_degree=12, amp='f16')
    neuron_model.to_neuron()
    return neuron_model

def infer(prompt):
    tokenizer = AutoTokenizer.from_pretrained('Llama-2-13b')
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # run inference with top-k sampling
    with torch.inference_mode():
        start = time.time()
        generated_sequences = neuron_model.sample(input_ids, sequence_length=2048, top_k=50)
        elapsed = time.time() - start

    generated_sequences = [tokenizer.decode(seq) for seq in generated_sequences]
    result = "generated sequences: " + str(generated_sequences) + " in " + str(elapsed) + " seconds"
    return {'result': result}

def handle(inputs: Input):
    global neuron_model
    if not neuron_model:
        neuron_model = load_model(inputs.get_properties())

    if inputs.is_empty():
        # Model server makes an empty call to warmup the model on startup
        return None

    data = inputs.get_as_json()
    prompt = data.pop("prompt")
    outputs = infer(prompt)
    result = {"outputs": outputs}
    return Output().add_as_json(result)