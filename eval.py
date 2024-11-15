from model.qwen2 import Qwen2VL

import argparse

def main(args):
    if args.model_name == "Qwen/Qwen2-VL-2B-Instruct":
        model = Qwen2VL(quantization_mode=args.quantization_mode)
    else:
        raise ValueError(f"Model {args.model_name} not supported")

    if args.benchmark_name == "docvqa":
        from benchmark.docvqa import DocVQA
        benchmark = DocVQA(model)
    else:
        raise ValueError(f"Benchmark {args.benchmark_name} not supported")

    benchmark.evaluate()
    benchmark.results()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--quantization_mode", type=int, default=16)
    args.add_argument("--batch_size", type=int, default=1)
    args.add_argument("--benchmark_name", type=str, default="docvqa")
    args.add_argument("--model_name", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    
    args = args.parse_args()

    main(args)