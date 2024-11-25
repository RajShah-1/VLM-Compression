# import os
# from PIL import Image
# from datasets import load_dataset

# # Load the dataset
# dataset = load_dataset("nielsr/docvqa_1200_examples", split="train")

# # Directory to save images (current directory)
# output_dir = os.getcwd()

# # Ensure the output directory exists
# os.makedirs(output_dir, exist_ok=True)

# # Save images as <id>.jpeg
# for example in dataset:
#     image_id = example["id"]
#     image_data = example["image"]  # Assuming this is PIL image data

#     # Save the image with the id as the filename
#     output_path = os.path.join(output_dir, f"{image_id}.jpeg")
#     image_data.save(output_path, "JPEG")

#     print(f"Saved: {output_path}")

import json
from datasets import load_dataset

output_file = "test.jsonl"

dataset = load_dataset("nielsr/docvqa_1200_examples", split="test")

with open(output_file, "a") as f:
    for example in dataset:
        image = "/home/hice1/drauthan3/scratch/BDA_project/prune/images" + example["id"] + ".jpeg"
        text = "The following is a conversation between a curious human and AI assistant. The assistant gives a direct answer in maximum two words to the user's question.\nHuman: <image>"
        
        for ans in example["answers"]:
            text += "\nHuman: " + example["query"]["en"] + "\nAI: " + ans
        
        task_type = "llava_sft"
        
        json_object = {
            "image": image,
            "text": text,
            "task_type": task_type
        }
        
        f.write(json.dumps(json_object) + "\n")