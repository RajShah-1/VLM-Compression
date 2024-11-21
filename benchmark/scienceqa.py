from datasets import load_dataset
from tqdm import tqdm
import os

class ScienceQA:
    def __init__(self, model):
        self.model = model
        self.processor = model.get_processor()
        self.model_type = self.model.get_model_name()
        self.cache_dir = os.path.join("datasets", "scienceqa")
        self.dataset = load_dataset("derek-thomas/ScienceQA", split="test", cache_dir=self.cache_dir)
        self.answers_unique = []
        self.generated_texts_unique = []

    def evaluate(self):
        EVAL_BATCH_SIZE = 1

        for i in tqdm(range(0, len(self.dataset), EVAL_BATCH_SIZE)):
            examples = self.dataset[i: i + EVAL_BATCH_SIZE]

            # Ensure examples is iterable
            if not isinstance(examples, list):
                examples = [examples]

            # Skip questions without images
            valid_examples = [
                example for example in examples if example["image"] and example["image"][0] is not None
            ]

            if not valid_examples:
                #print(f"Skipping batch {i} as no valid examples with images found.", flush=True)
                continue

            # Process valid examples
            self.answers_unique.extend([
                chr(ord('A') + int(example["answer"][0])) for example in valid_examples
            ])

            # Prepare inputs
            images = [
                [example["image"][0]] for example in valid_examples
            ]
            questions = [{"en": self._format_question(example)} for example in valid_examples]

            # Run model inference
            outputs = self.model.process_image_queries(images, questions)

            # Store generated texts
            self.generated_texts_unique.extend(outputs)
    def _format_question(self, example):
            """Format question with options"""
            question = example.get("question", "")
            for i, choice in enumerate(example.get("choices", [])):
                question += f"\n{chr(ord('A') + i)}. {choice}"
            question += "\nPlease answer directly with only the letter of the correct option."
            return question

    def results(self):
        # Clean outputs for comparison
        self.generated_texts_unique = [g.strip().upper() for g in self.generated_texts_unique]  # Ensure consistency
        
        # Calculate accuracy
        correct = sum(
            1 for pred, truth in zip(self.generated_texts_unique, self.answers_unique) if pred == truth
        )
        total = len(self.answers_unique)
        accuracy = correct / total * 100
        
        # Print accuracy
        print(f"Accuracy: {accuracy:.2f}% ({correct}/{total})")

        return accuracy