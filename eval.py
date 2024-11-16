from model.qwen2 import Qwen2VL
from model.idefics2 import Idefics2
from model.phi3 import Phi3_5
from model.llava import VideoLLava
from model.mplug import Mplug
# setup cli args
import argparse

def main(args):
    # Load the model based on the input argument
    if args.model_name == "Qwen/Qwen2-VL-2B-Instruct":
        model = Qwen2VL(quantization_mode=args.quantization_mode)
    elif args.model_name == "HuggingFaceM4/idefics2-8b":
        model = Idefics2(quantization_mode=args.quantization_mode)
    elif args.model_name == "microsoft/Phi-3.5-vision-instruct":
        model = Phi3_5(quantization_mode=args.quantization_mode)
    elif args.model_name == "LanguageBind/Video-LLaVA-7B-hf":
        model = VideoLLava(quantization_mode=args.quantization_mode)
    elif args.model_name == "LanguageBind/Video-LLaVA-7B-hf":
        model = VideoLLava(quantization_mode=args.quantization_mode)
    else:
        raise ValueError(f"Model {args.model_name} not supported")

    # Load the benchmark based on the input argument
    if args.benchmark_name == "docvqa":
        from benchmark.docvqa import DocVQA
        benchmark = DocVQA(model)
    elif args.benchmark_name == "vqa2":
        from benchmark.vqa2 import VQA_v2
        benchmark = VQA_v2(model)
    elif args.benchmark_name == "scienceqa":
        from benchmark.scienceqa import ScienceQA
        benchmark = ScienceQA(model)
    else:
        raise ValueError(f"Benchmark {args.benchmark_name} not supported")

    # Run the evaluation and display results
    benchmark.evaluate()
    benchmark.results()

if __name__ == "__main__":
    # Set up command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization_mode", type=int, default=4, help="Quantization mode for the model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--benchmark_name", type=str, default="scienceqa", help="Benchmark name to run (e.g., scienceqa, vqa2).")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct", help="Model name to evaluate.")

    args = parser.parse_args()

    main(args)
