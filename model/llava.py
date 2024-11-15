import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, VideoLlavaProcessor, VideoLlavaForConditionalGeneration, AutoTokenizer
from model.model import Model
from model.utils import setup_cache_dir

def get_model_tokenizer_processor(quantization_mode):
    model_name = "LanguageBind/Video-LLaVA-7B-hf"

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
        model = VideoLlavaForConditionalGeneration.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir
        )
    else:
        model = VideoLlavaForConditionalGeneration.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            cache_dir=cache_dir
        )

    processor = AutoProcessor.from_pretrained(model_name)

    return model, tokenizer, processor

class VideoLLava(Model):
    def __init__(self, quantization_mode):
        self.model, self.tokenizer, self.processor = get_model_tokenizer_processor(quantization_mode)
        self.processor.patch_size = self.model.config.vision_config.patch_size
        self.processor.vision_feature_select_strategy = self.model.config.vision_feature_select_strategy


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
        return "video_llava"

    def process_image_queries(self, images, queries):
        self.model.eval()
        
        torch.cuda.empty_cache()
        
        image = images[0][0]
        assert len(images) == 1, "Only one image is supported"
        assert len(queries) == 1, "Only one query is supported"
        q = queries[0]
        prompt = [
            f"USER: Answer briefly. <image> {q['en']} ASSISTANT:"
        ]

        inputs = self.processor(text=prompt, images=[image], padding=True, return_tensors="pt")
        inputs = inputs.to("cuda")
        generate_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_texts = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        generated_texts = [g.split("ASSISTANT:")[1].strip() for g in generated_texts]
        return generated_texts


    def process_generate(self, original_texts, images):
        self.model.eval()
        
        torch.cuda.empty_cache()
        

        texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
            for msg in messages
        ]

        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=texts,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )

        inputs = inputs.to("cuda")
        generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_texts

    def get_processor(self):
        return self.processor