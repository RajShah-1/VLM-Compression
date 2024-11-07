import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from transformers import BitsAndBytesConfig
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import os
from huggingface_hub import snapshot_download

def download_model():
    """
    Download the Qwen2-VL model files
    """
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    cache_dir = "./qwen2_vl_model"
    
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
        
    snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_dir=cache_dir
    )
    return cache_dir

def load_model_8bit():
    """
    Load Qwen2-VL model with 8-bit quantization
    """
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    
    # Configure 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )
    
    # First load the configuration
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # Then load the model with the correct class
    try:
        # Try loading with auto class first
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    except ValueError:
        # If that fails, try importing the specific model class
        from transformers import Qwen2ForCausalLM
        model = Qwen2ForCausalLM.from_pretrained(
            model_id,
            config=config,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    
    return model

def load_model_4bit():
    """
    Load Qwen2-VL model with 4-bit quantization
    """
    model_id = "Qwen/Qwen2-VL-2B-Instruct"
    
    # Configure 4-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    # First load the configuration
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    
    # Then load the model with the correct class
    try:
        # Try loading with auto class first
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            config=config,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    except ValueError:
        # If that fails, try importing the specific model class
        from transformers import Qwen2ForCausalLM
        model = Qwen2ForCausalLM.from_pretrained(
            model_id,
            config=config,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    
    return model

def convert_to_mlx(model, save_path):
    """
    Convert the quantized PyTorch model to MLX format
    """
    def convert_layer(layer, state_dict, prefix=''):
        """Helper function to convert transformer layers"""
        if hasattr(layer, 'weight'):
            key = f"{prefix}.weight"
            # Convert weight to MLX array
            weight = layer.weight.detach().cpu().numpy()
            state_dict[key] = mx.array(weight)
            
        if hasattr(layer, 'bias') and layer.bias is not None:
            key = f"{prefix}.bias"
            bias = layer.bias.detach().cpu().numpy()
            state_dict[key] = mx.array(bias)
    
    mlx_state_dict = {}
    
    # Convert each layer
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm, torch.nn.Embedding)):
            convert_layer(module, mlx_state_dict, name)
    
    # Save the converted weights
    mx.save(save_path, mlx_state_dict)
    print(f"Model saved to {save_path}")
    
    return mlx_state_dict

def verify_mlx_model(mlx_state_dict, save_path):
    """
    Verify the saved MLX model
    """
    # Load the saved model and compare
    loaded_state_dict = mx.load(save_path)
    
    # Check if all keys match
    original_keys = set(mlx_state_dict.keys())
    loaded_keys = set(loaded_state_dict.keys())
    
    if original_keys == loaded_keys:
        print("✓ Model verification successful: All layers converted correctly")
    else:
        print("⚠ Model verification failed: Mismatched layers")
        print("Missing keys:", original_keys - loaded_keys)
        print("Extra keys:", loaded_keys - original_keys)

def main(quantization_bits=8, output_path='qwen_vl_mlx.npz'):
    """
    Main function to orchestrate the quantization and conversion process
    """
    print(f"Starting Qwen2-VL-2B-Instruct quantization ({quantization_bits}-bit)...")
    
    # 1. Download model if needed
    print("Downloading/Loading model...")
    cache_dir = download_model()
    
    # 2. Load and quantize model
    if quantization_bits == 8:
        model = load_model_8bit()
    else:
        model = load_model_4bit()
    
    print("Model loaded and quantized successfully")
    
    # 3. Convert to MLX
    print("Converting to MLX format...")
    mlx_state_dict = convert_to_mlx(model, output_path)
    
    # 4. Verify conversion
    print("Verifying converted model...")
    verify_mlx_model(mlx_state_dict, output_path)
    
    return mlx_state_dict

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Quantize Qwen2-VL model and convert to MLX')
    parser.add_argument('--bits', type=int, choices=[4, 8], default=8,
                        help='Quantization precision (4 or 8 bits)')
    parser.add_argument('--output', type=str, default='qwen_vl_mlx.npz',
                        help='Output path for the MLX model')
    
    args = parser.parse_args()
    main(quantization_bits=args.bits, output_path=args.output)