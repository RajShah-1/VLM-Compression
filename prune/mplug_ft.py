import os
import torch
import sys

from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
import os
import sys
mplug_owl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mPLUG-Owl'))
sys.path.append(mplug_owl_path)
from mplug_owl.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from PIL import Image
import random
import json


# Function to create the dataset
def create_dataset():
    train_dataset = load_dataset('nielsr/docvqa_1200_examples', split='train')
    train_dataset = train_dataset.remove_columns(['id', 'words', 'bounding_boxes', 'answer'])

    eval_dataset = load_dataset('nielsr/docvqa_1200_examples', split='test')
    eval_dataset = eval_dataset.remove_columns(['id', 'words', 'bounding_boxes', 'answer']) # only image, query and answers present

    return train_dataset, eval_dataset


# Custom data collator
class MPlugDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        assert len(examples) == 1, 'Batch size must be 1 for MPlug models'
        example = examples[0]

        # image = Image.open(example['image']) if isinstance(example['image'], str) else example['image']
        image = example['image']
        question = example['query']['en']
        answer = random.choice(example['answers'])
        # prompt_message = [f"The following is a conversation between a curious human and AI assistant. The assistant gives a direct answer in maximum two words to the user's question.\nHuman: <image>\nHuman: {question}\nAI:"]
        prompt_message = [f'''The following is a conversation between a curious human and AI assistant. The assistant gives a direct answer in maximum two words to the user's question.
                          Human: <image>
                          Human: {question}
                          AI:''']

        # Process input and labels
        # batch = self.processor(prompt_message, [image], return_tensors='pt', padding=True)
        batch = self.processor(text=prompt_message, images=[image], return_tensors='pt')
        prompt_input_ids = batch['input_ids']
        answer_input_ids = self.processor.tokenizer(
            answer, add_special_tokens=False, return_tensors='pt'
        )['input_ids']
        print("prompt_input_ids shape:", prompt_input_ids.shape)
        print("answer_input_ids shape:", answer_input_ids.shape)
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
        batch['labels'] = labels
        del batch['attention_mask']

        return batch


# Setup cache directory
def setup_cache_dir():
    cache_dir = os.path.join(os.getcwd(), "model_cache")
    os.makedirs(cache_dir, exist_ok=True)
    os.environ['TRANSFORMERS_CACHE'] = cache_dir
    os.environ['HF_HOME'] = cache_dir
    os.environ['HF_DATASETS_CACHE'] = os.path.join(cache_dir, "datasets")
    return cache_dir

# Evaluation function
def evaluate_model(model, processor, eval_dataset, output_dir):
    print("Starting evaluation...")
    generated_texts = []
    answers = []

    for example in eval_dataset:
        image = Image.open(example['image']) if isinstance(example['image'], str) else example['image']
        question = example['query']['en']
        answers.append(example['answers'])

        prompts = [
            f'''The following is a conversation between a curious human and AI assistant. The assistant gives a direct answer in maximum two words to the user's question.
            Human: <image>
            Human: {question}
            AI: '''
        ]
        inputs = processor(prompt_message, [image], return_tensors='pt').to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=64)
        generated_text = processor.tokenizer.batch_decode(
            generated_ids[:, inputs['input_ids'].size(1):],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        generated_texts.append(generated_text)

    # Save evaluation results
    evaluation_path = os.path.join(output_dir, "evaluation_results.json")
    with open(evaluation_path, 'w') as f:
        results = {"answers": answers, "generated_texts": generated_texts}
        json.dump(results, f, indent=4)

    print(f"Evaluation results saved to {evaluation_path}")
    
# Main function
def main():
    # Set model and cache directories
    model_name = "MAGAer13/mplug-owl-llama-7b"
    output_dir = os.path.join(os.getcwd(), "mplug_output")
    cache_dir = setup_cache_dir()

    # Load processor and model
    image_processor = MplugOwlImageProcessor.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    processor = MplugOwlProcessor(image_processor, tokenizer)
    model = MplugOwlForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        cache_dir=cache_dir
    ).to("cuda")

    # Apply LoRA if specified
    # use_lora = True
    # if use_lora:
    #     peft_config = LoraConfig(
    #         target_modules=[".*language_model.*\.(q_proj|v_proj)"],
    #         inference_mode=False,
    #         r=8,
    #         lora_alpha=32,
    #         lora_dropout=0.1,
    #     )
    #     model = get_peft_model(model, peft_config)
    #     model.print_trainable_parameters()

    # Create datasets
    train_dataset, eval_dataset = create_dataset()

    # Define training arguments
    training_args = TrainingArguments(
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_checkpointing=True,
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
        save_strategy="epoch",
    )

    # Initialize Trainer
    data_collator = MPlugDataCollator(processor)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    # Train the model
    trainer.train()

    # Save the model and processor
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    print(f"Model and processor saved to {output_dir}")

    # Evaluate the model
    evaluate_model(model, processor, eval_dataset, output_dir)


if __name__ == '__main__':
    main()