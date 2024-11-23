import torch
import os

from transformers import AutoTokenizer, BitsAndBytesConfig,  AutoProcessor, AutoModel
from icecream import ic
from model.model import Model
from model.utils import setup_cache_dir
from PIL import Image

import time

def get_model_tokenizer_processor(quantization_mode):
    model_name = "mPLUG/DocOwl2"

    if quantization_mode == 8:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    elif quantization_mode == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    else:
        # load the default model 
        quantization_config = None

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    cache_dir = setup_cache_dir()

    if quantization_config is not None:
        model = AutoModel.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir
        )
    else:
        model = AutoModel.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir
        )

    processor = AutoProcessor.from_pretrained("mPLUG/DocOwl2")

    return model, tokenizer, processor

class MplugDoc(Model):
    def __init__(self, quantization_mode):
        self.model, self.tokenizer = get_model_tokenizer_processor(quantization_mode)
        
        self.quantization_mode = quantization_mode
        self.num_processed = 0
        self.total_processing_time = 0

    def get_average_processing_time(self):
        if self.num_processed == 0:
            return 0
        return self.total_processing_time / self.num_processed
    
    def get_model_name(self):
        return f"MplugDoc_{self.quantization_mode}bit"
    
    def generate(self, texts, images):
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_texts = self.processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
        return generated_texts
    
    def process_image_queries(self, images, queries):
        torch.cuda.empty_cache()
        start_time = time.time()
        
        prompts = [{'role': 'USER', 'content': '<|image|>'*len(images)+query} for query in queries]

        output = self.process_generate(messages)
        time_end = time.time()

        self.total_processing_time += time_end - time_start
        self.num_processed += 1

        return output


    def get_processor(self):
        return self.processor