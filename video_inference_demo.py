from model.qwen2 import Qwen2VL
from model.idefics2 import Idefics2
from model.phi3 import Phi3_5
from model.llava import VideoLLava
from model.llavanext import LlavaNext

import argparse
import pandas as pd

from datetime import datetime
import os


def main(args):
    video_path = args.video_path
    if args.model_name == "LanguageBind/Video-LLaVA-7B-hf":
        model = VideoLLava(quantization_mode=args.quantization_mode)
    elif args.model_name == "Qwen/Qwen2-VL-2B-Instruct":
        model = Qwen2VL(quantization_mode=args.quantization_mode)
    elif args.model_name == "microsoft/Phi-3.5-vision-instruct":
        model = Phi3_5(quantization_mode=args.quantization_mode)
    elif args.model_name == "HuggingFaceM4/idefics2-8b":
        model = Idefics2(quantization_mode=args.quantization_mode)
    elif args.model_name == "llava-hf/LLaVA-NeXT-Video-7B-hf":
        model = LlavaNext(quantization_mode=args.quantization_mode)
    return model.video_inference(video_path=video_path, user_query="What is happening in the video ?")


if __name__ == "__main__":
    print("Running video inference demo")
    parser = argparse.ArgumentParser()
    parser.add_argument("--quantization_mode", type=int, default=16, help="Quantization mode for the model.")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation.")
    parser.add_argument("--benchmark_name", type=str, default="docvqa", help="Benchmark name to run (e.g., scienceqa, vqa2).")
    parser.add_argument("--model_name", type=str, default="HuggingFaceM4/idefics2-8b", help="Model name to evaluate.")
    parser.add_argument("--video_path", type=str, default="demo_video/sample_demo_1.mp4", help="Path to the video file")

    args = parser.parse_args()

    # quantization_modes = [4,8, 32]
    quantization_modes = [args.quantization_mode]

    output_dic = {}

    for quantization_mode in quantization_modes:
        args.quantization_mode = quantization_mode
        print("Running for quantization mode: ", quantization_mode)
        output = main(args)

        print(output)

        output_dic[quantization_mode] = output
    
    print("Output dictionary: ", output_dic)
