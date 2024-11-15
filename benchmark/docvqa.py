from datasets import load_dataset
from tqdm import tqdm
import torch
from peft import LoraConfig
from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration, Qwen2VLModel, Qwen2VLConfig
from qwen_vl_utils import process_vision_info
from model.model import Model
from benchmark.benchmark import Benchmark
from model.utils import setup_cache_dir
import base64
from io import BytesIO
from benchmark.utils import average_normalized_levenshtein_similarity

import os

class DocVQA(Benchmark):
    def __init__(self, model : Model):
        self.model = model
        self.processor = model.get_processor()
        self.cache_dir = os.path.join(setup_cache_dir(), "datasets")
        self.dataset = load_dataset("nielsr/docvqa_1200_examples", split="test", cache_dir=self.cache_dir)
        self.dataset = self.dataset.remove_columns(['id', 'words', 'bounding_boxes', 'answer'])
        self.answers_unique = []
        self.generated_texts_unique = []

    def evaluate(self):
        EVAL_BATCH_SIZE = 4

        for i in tqdm(range(0, len(self.dataset), EVAL_BATCH_SIZE)):
            examples = self.dataset[i: i + EVAL_BATCH_SIZE]
            self.answers_unique.extend(examples["answers"])
            images = [[im] for im in examples["image"]]
            print("Images inside is ", images, flush=True)
            # print("Examples queries are ", examples["query"], flush=True)
            texts = []
            for idx, q in enumerate(examples["query"]):
                print(q)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Answer briefly."},
                            {"type": "image", "image": images[idx][0], "resized_height": 280,"resized_width": 420},
                            {"type": "text", "text": q["en"]}
                        ]
                    }
                ]
                
                texts.append(messages)
            print(texts)
            output = self.model.process_generate(texts)
            print("Output is ", output, flush=True)
            print("Expected is ", examples["answers"], flush=True)
            self.generated_texts_unique.extend(output)
    
    def results(self):
        self.generated_texts_unique = [g.strip().strip(".") for g in self.generated_texts_unique]
        anls = average_normalized_levenshtein_similarity(
            ground_truth=self.answers_unique, predicted_answers=self.generated_texts_unique,
        )
        print(anls)