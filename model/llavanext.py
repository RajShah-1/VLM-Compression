from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration, AutoTokenizer
import torch
from PIL import Image
import requests
from model.model import Model
from model.utils import setup_cache_dir,read_video_pyav
import time

def get_model_tokenizer_processor(quantization_mode):
    model_name = "llava-hf/llava-v1.6-mistral-7b-hf"

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
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir
        )
    else:
        model = LlavaNextForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir
        )
    
    processor = LlavaNextProcessor.from_pretrained(model_name)

    return model, tokenizer, processor

class LlavaNext(Model):
    def __init__(self, quantization_mode):
        self.model, self.tokenizer, self.processor = get_model_tokenizer_processor(quantization_mode)
        self.quantization_mode = quantization_mode
        self.num_processed = 0
        self.total_processing_time = 0
        
    def get_average_processing_time(self):
        if self.num_processed == 0:
            return 0
        return self.total_processing_time
    
    def process(self, texts, images):
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)
        return outputs
    
    def generate(self, texts, images):
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_texts = self.processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)
        return generated_texts

    def get_model_name(self):
        return f"llavanext_{self.quantization_mode}bit"
    
    def process_image_queries(self, images, queries):
        start_time = time.time()
        self.model.eval()
        
        torch.cuda.empty_cache()
        
        texts = []
        for q in queries:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer briefly."},
                        {"type": "image"},
                        {"type": "text", "text": q["en"]}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            texts.append(text.strip())
        inputs = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=64)
        generated_texts = self.processor.batch_decode(generated_ids[:, inputs["input_ids"].size(1):], skip_special_tokens=True)

        end_time = time.time()
        self.total_processing_time += end_time - start_time
        self.num_processed += 1

        return generated_texts
    
    def process_generate(self, original_texts, images):
        pass

    def get_processor(self):
        return self.processor
        