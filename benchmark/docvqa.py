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
from benchmark.utils import average_normalized_levenshtein_similarity, average_bert_score_f1_value

import os

class DocVQA(Benchmark):
    def __init__(self, model : Model):
        self.model = model
        self.processor = model.get_processor()
        self.model_type = self.model.get_model_name()
        self.cache_dir = os.path.join(setup_cache_dir(), "datasets")
        self.dataset = load_dataset("nielsr/docvqa_1200_examples", split="test", cache_dir=self.cache_dir)
        self.dataset = self.dataset.remove_columns(['id', 'words', 'bounding_boxes', 'answer'])
        self.answers_unique = []
        self.generated_texts_unique = []

    def evaluate(self):
        EVAL_BATCH_SIZE = 1

        for i in tqdm(range(0, len(self.dataset), EVAL_BATCH_SIZE)):
            examples = self.dataset[i: i + EVAL_BATCH_SIZE]
            self.answers_unique.extend(examples["answers"])
            images = [[im] for im in examples["image"]]
            queries = examples["query"]
            print("Images inside is ", images, flush=True)
            
            output = self.model.process_image_queries(images, queries)
        
            print("Output is ", output, flush=True)
            print("Expected is ", examples["answers"], flush=True)
            self.generated_texts_unique.extend(output)
    
    def results(self):
        self.generated_texts_unique = [g.strip().strip(".") for g in self.generated_texts_unique]
        anls = average_normalized_levenshtein_similarity(
            ground_truth=self.answers_unique, predicted_answers=self.generated_texts_unique,
        )

        bert_score_f1 = average_bert_score_f1_value(
            ground_truth=self.answers_unique, predicted_answers=self.generated_texts_unique,
        )

        return anls, bert_score_f1