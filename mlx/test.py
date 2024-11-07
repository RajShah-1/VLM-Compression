import torch
import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer, CLIPImageProcessor
from transformers import Qwen2VLForCausalLM
from PIL import Image
import numpy as np

def convert_qwen_weights_to_mlx(hf_model):
    """Convert Huggingface Qwen weights to MLX format."""
    mlx_weights = {}
    
    for name, param in hf_model.named_parameters():
        # Convert the parameter to numpy first
        param_np = param.detach().float().cpu().numpy()
        
        # Handle different parameter types
        if 'visual' in name:
            # Vision encoder weights
            if 'weight' in name:
                param_np = param_np.astype(np.float32)
            mlx_weights[name] = mx.array(param_np)
        else:
            # Language model weights
            if param.dtype == torch.int8:
                # Dequantize 4-bit weights
                scale = param.scale.float().cpu().numpy()
                param_np = param_np.astype(np.float32) * scale
            mlx_weights[name] = mx.array(param_np)
    
    return mlx_weights

def load_and_convert_model(model_name):
    """Load model from Huggingface and convert to MLX."""
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Use CLIP image processor with Qwen2 settings
    image_processor = CLIPImageProcessor(
        do_resize=True,
        size={"height": 224, "width": 224},
        do_center_crop=True,
        crop_size={"height": 224, "width": 224},
        do_normalize=True,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711]
    )
    
    # Load the model using the specific Qwen2VL model class
    model = Qwen2VLForCausalLM.from_pretrained(
        model_name,
        device_map="cpu",
        torch_dtype=torch.float32
    )
    
    # Convert weights
    mlx_weights = convert_qwen_weights_to_mlx(model)
    
    return tokenizer, image_processor, mlx_weights, model.config

def process_image(image_processor, image_path):
    """Process image using the CLIP image processor."""
    image = Image.open(image_path)
    inputs = image_processor(
        images=image,
        return_tensors="pt"
    )
    
    # Convert image features to MLX array
    pixel_values = inputs.pixel_values.numpy()
    return mx.array(pixel_values)

def generate_response(tokenizer, image_features, prompt="Describe this image:"):
    """Prepare input for text generation."""
    # Tokenize the prompt
    text_inputs = tokenizer(prompt, return_tensors="pt")
    
    return {
        "input_ids": mx.array(text_inputs.input_ids.numpy()),
        "attention_mask": mx.array(text_inputs.attention_mask.numpy()),
        "image_features": image_features
    }

def main():
    # Model and image paths
    model_name = "ksukrit/qwen2-vl-2b-4bit"
    image_path = "cats.jpg"
    
    print("Loading model and processors...")
    tokenizer, image_processor, mlx_weights, model_config = load_and_convert_model(model_name)
    
    print("Processing image...")
    image_features = process_image(image_processor, image_path)
    
    # Prepare inputs for generation
    inputs = generate_response(tokenizer, image_features)
    
    print("Model conversion complete")
    print("MLX weights shape:", {k: v.shape for k, v in mlx_weights.items()})
    print("Image features shape:", image_features.shape)
    
    # Save MLX weights and config
    mx.save("qwen_mlx_weights.npz", mlx_weights)
    
    # Save important config parameters
    config_dict = {
        "hidden_size": model_config.hidden_size,
        "num_attention_heads": model_config.num_attention_heads,
        "num_hidden_layers": model_config.num_hidden_layers,
        "vocab_size": model_config.vocab_size,
        "vision_config": {
            "hidden_size": model_config.vision_config.hidden_size,
            "num_attention_heads": model_config.vision_config.num_attention_heads,
            "num_hidden_layers": model_config.vision_config.num_hidden_layers,
        }
    }
    np.save("qwen_config.npy", config_dict)
    
    print("Weights saved to qwen_mlx_weights.npz")
    print("Config saved to qwen_config.npy")

if __name__ == "__main__":
    main()