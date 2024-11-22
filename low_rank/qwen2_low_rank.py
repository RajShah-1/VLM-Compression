import os
import torch
from torch import nn
import torch.nn.utils.prune as prune
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoProcessor, Trainer, TrainingArguments
from qwen_vl_utils import process_vision_info

from model.utils import setup_cache_dir

import random
import json

class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank_ratio):
        super(LowRankLinear, self).__init__()
        self.rank = max(1, int(rank_ratio * min(in_features, out_features)))
        self.in_features = in_features
        self.out_features = out_features
        
        self.linear1 = nn.Linear(in_features, self.rank, bias=False)
        self.linear2 = nn.Linear(self.rank, out_features, bias=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

    @property
    def weight(self):
        # Mimic the weight attribute by combining the two linear layers
        # FIXME Come up with a better way to handle this
        return self.linear2.weight.data @ self.linear1.weight.data
    
    @classmethod
    def from_linear(cls, linear_layer, rank_ratio):
        """
        Initialize from an existing nn.Linear layer using SVD decomposition
        with proper dimension handling
        """
        device = linear_layer.weight.device
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        
        instance = cls(in_features, out_features, rank_ratio)
        instance = instance.to(device)
        rank     = instance.rank

        # Init low rank layers using SVD
        with torch.no_grad():
            try:
                U, S, V = torch.svd(linear_layer.weight.data)
                sqrt_s = torch.sqrt(S[:rank]).view(-1, 1)
                
                # Expected dimensions:
                # linear1.weight -> (rank, in_features)
                # linear2.weight -> (out_features, rank)
                instance.linear1.weight.data = (V[:, :rank] * sqrt_s.T).T
                instance.linear2.weight.data = U[:, :rank] * sqrt_s.T
                
                # Handle bias at 2nd layer
                if linear_layer.bias is not None:
                    instance.linear2.bias.data.copy_(linear_layer.bias.data)
                    
                # Sanity check
                assert instance.linear1.weight.shape == (rank, in_features), \
                    f"Linear1 weight shape mismatch: got {instance.linear1.weight.shape}, expected {(rank, in_features)}"
                assert instance.linear2.weight.shape == (out_features, rank), \
                    f"Linear2 weight shape mismatch: got {instance.linear2.weight.shape}, expected {(out_features, rank)}"
                
            except Exception as e:
                raise RuntimeError(f"SVD initialization failed: {str(e)}\n"
                                 f"Shapes: weight={linear_layer.weight.shape}, "
                                 f"U={U.shape}, S={S.shape}, V={V.shape}, "
                                 f"rank={rank}")
            
        return instance
    
    def get_compression_stats(self):
        """
        Return compression statistics
        """
        original_params = self.in_features * self.out_features + self.out_features
        compressed_params = (self.in_features * self.rank +    # first layer weights
                             self.rank * self.out_features +   # second layer weights
                             self.out_features)                # bias
        
        stats = {
            'original_params': original_params,
            'compressed_params': compressed_params,
            'compression_ratio': compressed_params / original_params,
            'rank': self.rank,
        }
        return stats

def replace_linear_with_low_rank(model, rank_ratio, skip_patterns=None):
    """
    Replace linear layers with low-rank versions initialized using SVD
    """
    if skip_patterns is None:
        skip_patterns = []
        
    replacements = 0
    total_params_before = 0
    total_params_after = 0
    
    for name, module in model.named_modules():
        if any(pattern in name for pattern in skip_patterns):
            continue
            
        if isinstance(module, nn.Linear):
            try:
                print(f"\nProcessing layer {name}")
                print(f"Original shape: {module.weight.shape}")
                
                low_rank_module = LowRankLinear.from_linear(module, rank_ratio)                
                print(f"Low rank shapes: {low_rank_module.linear1.weight.shape} -> {low_rank_module.linear2.weight.shape}")
                
                # Compression stats
                stats = low_rank_module.get_compression_stats()
                total_params_before += stats['original_params']
                total_params_after += stats['compressed_params']
                
                # Replace the module
                parent_module = model
                components = name.split('.')
                for comp in components[:-1]:
                    parent_module = getattr(parent_module, comp)
                setattr(parent_module, components[-1], low_rank_module)
                
                replacements += 1
                
                # Print retained variance
                with torch.no_grad():
                    U, S, V = torch.svd(module.weight.data)
                    rank = low_rank_module.rank
                    total_variance = torch.sum(S ** 2)
                    retained_variance = torch.sum(S[:rank] ** 2)
                    variance_ratio = retained_variance / total_variance
                    print(f"Layer {name}: Retaining {variance_ratio:.2%} of variance")
                    
            except Exception as e:
                print(f"Warning: Could not replace layer {name}: {e}")
                continue
            
    compression_ratio = total_params_after / total_params_before
    print(f"\nSummary:")
    print(f"Replaced {replacements} linear layers")
    print(f"Parameters before: {total_params_before:,}")
    print(f"Parameters after: {total_params_after:,}")
    print(f"Compression ratio: {compression_ratio:.2%}")
    
    return model

# Function to create the dataset
def create_dataset():
    cache_dir = setup_cache_dir()
    train_dataset = load_dataset('nielsr/docvqa_1200_examples', split='train')
    train_dataset = train_dataset.remove_columns(['id', 'words', 'bounding_boxes', 'answer'])

    eval_dataset = load_dataset('nielsr/docvqa_1200_examples', split='test')
    eval_dataset = eval_dataset.remove_columns(['id', 'words', 'bounding_boxes', 'answer'])

    return train_dataset, eval_dataset

class MiniDocVQADataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        assert len(examples) == 1, 'Batch size must be 1!'
        example = examples[0]

        image = example['image']
        question = example['query']['en']
        answer = random.choice(example['answers'])
        prompt_message = {
            'role': 'user',
            'content': f'<|image_1|>\n{question}\nAnswer briefly.',
        }
        
        # Qwen specific step
        image_inputs, video_inputs = process_vision_info([prompt_message])

        prompt = self.processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )
        answer = f'{answer}<|end|>\n<|endoftext|>'

        # Process input and labels
        batch = self.processor(text=prompt, images=image_inputs, videos=video_inputs, return_tensors='pt', padding=True)
        prompt_input_ids = batch['input_ids']
        answer_input_ids = self.processor.tokenizer(
            answer, add_special_tokens=False, return_tensors='pt'
        )['input_ids']
        input_ids = torch.cat([prompt_input_ids, answer_input_ids], dim=1)

        ignore_index = -100
        labels = torch.cat(
            [
                torch.tensor([ignore_index] * len(prompt_input_ids[0])).unsqueeze(0),
                answer_input_ids,
            ],
            dim=1,
        )

        batch['input_ids'] = input_ids
        del batch['attention_mask']
        batch['labels'] = labels

        return batch


def evaluate_model(model):
    from benchmark.docvqa import DocVQA
    from datetime import datetime
    import pandas as pd

    benchmark = DocVQA(model)
    benchmark.evaluate()
    result = benchmark.results()

    print(f"Model: {model.get_model_name()}, Benchmark: DocVQA, Accuracy: {result}")
    now = datetime.now()

    df = pd.DataFrame(columns=["timestamp", "model_name", "benchmark", "accuracy", "memory_utilization", "model_runtime", "additional_results"])
    formatted_timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
    df = pd.concat([df, pd.DataFrame([[formatted_timestamp, model.get_model_name(), 'DocVQA', result, model.get_model_size(), model.get_average_processing_time() , ""]], columns=df.columns)], ignore_index=True)
    df.to_csv("results.csv", index=False)

def get_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.numel() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.numel() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024 ** 2
    return size_all_mb


def main():
    model_name = "Qwen/Qwen2-VL-2B-Instruct"
    cache_dir = setup_cache_dir()
    output_dir = os.path.join(os.getcwd(), "qwen2_low_rank")

    from model.qwen2 import Qwen2VL, CustomQwen2VL

    qwen2 = Qwen2VL(quantization_mode=None)
    model, tokenizer, processor = qwen2.model, qwen2.tokenizer, qwen2.processor

    # Prune the model
    num_layers = len(model.model.layers)  # Total number of layers
    print(f"Original number of layers: {num_layers}")
    # Actual pruning code would come here
    # ===
    print("Original model size:", get_model_size(model))
    
    # Replace with low-rank layers, using SVD initialization
    model = replace_linear_with_low_rank(
        model, 
        rank_ratio=0.5,
        skip_patterns=['embedding', 'final_layer']
    )
    
    print("Compressed model size:", get_model_size(model))
    # ===

    # Prepare datasets for training
    train_dataset = load_dataset('nielsr/docvqa_1200_examples', split='train', cache_dir=cache_dir)
    eval_dataset = load_dataset('nielsr/docvqa_1200_examples', split='test', cache_dir=cache_dir)

    # Define training arguments
    training_args = TrainingArguments(
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_checkpointing=False,
        optim='adamw_torch',
        learning_rate=4e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        lr_scheduler_type='linear',
        warmup_steps=50,
        logging_steps=10,
        output_dir=output_dir,
        bf16=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        dataloader_prefetch_factor=2,
        save_strategy="no",
    )

    data_collator = MiniDocVQADataCollator(processor)

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    # Fine-tune the model
    print("\nStarting fine-tuning...\n")
    trainer.train()
    print("\nFine-tuning completed.")

    os.makedirs(output_dir, exist_ok=True)

    print("Saving model with device_map='auto' for offloading...")
    model.save_pretrained(output_dir, safe_serialization=False, device_map="auto")
    print("Model saved successfully.")

    print("Saving processor")
    if not hasattr(processor, 'chat_template'):
        processor.chat_template = None

    print("Saving processor...")
    processor.save_pretrained(output_dir)
    print(f"Processor saved to {output_dir}")

    torch.save(model, f"{output_dir}/modified_model.pth")

    custom_model = CustomQwen2VL(None, model, tokenizer, processor)
    # Or custom_model could be loaded as following:
    # custom_model = CustomQwen2VL.from_path(output_dir)
    evaluate_model(custom_model)
    print("Training and evaluation complete.")

if __name__ == "__main__":
    main()
