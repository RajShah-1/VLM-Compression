from model.qwen2 import Qwen2VL
from model.idefics2 import Idefics2
from model.phi3 import Phi3_5
from model.llava import VideoLLava

# setup cli args
import argparse

def main(args):
    if args.model_name == "Qwen/Qwen2-VL-2B-Instruct":
        model = Qwen2VL(quantization_mode=args.quantization_mode)
    elif args.model_name == "HuggingFaceM4/idefics2-8b":
        model = Idefics2(quantization_mode=args.quantization_mode)
    elif args.model_name == "microsoft/Phi-3.5-vision-instruct":
        model = Phi3_5(quantization_mode=args.quantization_mode)
    elif args.model_name == "LanguageBind/Video-LLaVA-7B-hf":
        model = VideoLLava(quantization_mode=args.quantization_mode)
    else:
        raise ValueError(f"Model {args.model_name} not supported")

    if args.benchmark_name == "docvqa":
        from benchmark.docvqa import DocVQA
        benchmark = DocVQA(model)
    elif args.benchmark_name == "vqa2":
        from benchmark.vqa2 import VQA_v2
        benchmark = VQA_v2(model)
    else:
        raise ValueError(f"Benchmark {args.benchmark_name} not supported")

    benchmark.evaluate()
    benchmark.results()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--quantization_mode", type=int, default=4)
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--benchmark_name", type=str, default="docvqa")
    args.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    
    args = args.parse_args()

    main(args)