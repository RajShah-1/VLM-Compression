import torch
from transformers import AutoTokenizer, BitsAndBytesConfig
import logging
from transformers import Qwen2VLModel, Qwen2VLConfig

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
    model_name="Qwen/Qwen2-VL-2B-Instruct",
    output_dir="./quantized_model",
):
    logger = setup_logging()
    logger.info(f"Starting quantization of {model_name}")

    try:
        # Load tokenizer
        logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Configure 4-bit quantization
        logger.info("Configuring 4-bit quantization...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Load configuration
        logger.info("Loading model configuration...")
        config = Qwen2VLConfig.from_pretrained(
            model_name,
            trust_remote_code=True
        )

        # Load and quantize model
        logger.info("Loading and quantizing model...")
        model = Qwen2VLModel.from_pretrained(
            model_name,
            config=config,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )

        logger.info("Quantization completed successfully")

        # Save the quantized model
        logger.info(f"Saving quantized model to {output_dir}")
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

        logger.info("Model and tokenizer saved successfully")

        # Free up memory
        del model
        torch.cuda.empty_cache()

        return True

    except Exception as e:
        logger.error(f"Error during quantization: {str(e)}")
        return False

def main():
    # Set your parameters here
    params = {
        "model_name": "Qwen/Qwen2-VL-2B-Instruct",
        "output_dir": "./qwen2_vl_2b_4bit"
    }

    success = quantize_model(**params)

    if success:
        print("Quantization completed successfully!")
    else:
        print("Quantization failed. Check logs for details.")

if __name__ == "__main__":
    main()