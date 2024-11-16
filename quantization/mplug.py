import sys
import logging
import torch
import os
mplug_owl_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mPLUG-Owl'))
sys.path.append(mplug_owl_path)
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from transformers import AutoTokenizer, BitsAndBytesConfig
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('quantization.log')
        ]
    )
    return logging.getLogger(__name__)

def quantize_model(
    model_name="MAGAer13/mplug-owl-llama-7b-video",
    output_dir="./quantized_model",
):
    logger = setup_logging()
    logger.info(f"Starting quantization of {model_name}")
    
    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Configure 4-bit quantization
        logger.info("Configuring 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
        
        # Load configuration
        # logger.info("Loading model configuration...")
        # config = MplugOwlForConditionalGeneration.from_pretrained(
        #     model_name,
        #     device_map="auto",
        #     trust_remote_code=True
        # )
        
        # Load and quantize model
        logger.info("Loading and quantizing model...")
        model = MplugOwlForConditionalGeneration.from_pretrained(model_name, quantization_config=quantization_config, device_map="auto", trust_remote_code=True)
        
        logger.info("Quantization completed successfully")

        # Save the quantized model
        logger.info(f"Saving quantized model to {output_dir}")
        # model.save_pretrained(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
        tokenizer.save_pretrained(output_dir)
        logger.info(f"Quantized model footprint is {model.get_memory_footprint()}")

        logger.info("Model and tokenizer saved successfully")

        # Free up memory
        del model
        torch.cuda.empty_cache()
        
        original_model = MplugOwlForConditionalGeneration.from_pretrained(model_name, device_map="auto")
        logger.info(f"Original Model footprint is {original_model.get_memory_footprint()}")

        return True
    
    except Exception as e:
        logger.error(f"Error during quantization: {str(e)}")
        return False

def main():
    # Set your parameters here
    params = {
        "model_name": "MAGAer13/mplug-owl-llama-7b-video",
        "output_dir": "./mplug-owl-llama-7b-video"
    }

    success = quantize_model(**params)

    if success:
        print("Quantization completed successfully!")
    else:
        print("Quantization failed. Check logs for details.")

if __name__ == "__main__":
    main()