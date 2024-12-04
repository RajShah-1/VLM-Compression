# Requirements

The device must have 

1. Optional: Create and Activate virtualenv or conda env 
2. Install dependencies using `pip3 install requirements.txt`

# Directory Structure

1. `model/` Contains the logic for model loading, pruning, video and image inference. This also includes utility to get the memory footprint, processing time etc. Each custom model inherits from the `model/model.py` file. `utils.py` contains utility functions to process video files and setting up the cache correctly.
2. `benchmark/` Contains the benchmark like VQAv2, Flick30k, ScienceQA, DocVQA. This involves the data loading, calling the models, computing and returning the results. 
3. `eval.py`, `video_inference_demo.py`, `benchmark_launcher.py` : Entry points into our code, explained in more detail in the next section
4. `phi3_pruning_code/` : Contains the code to prune and FT the phi3.5 model on the 3 different datasets and compute the benchmark results
5. `low_rank/` : Contains the code to perform low-rank dense layer optimizations for the Qwen2VL model. See below section for how to run.

# How to Run:

## Quantized Models

### Running benchmarks and Evaluation of models

Run single benchmark for one particular model and quantization mode.

`python3 eval.py --quantization_mode 8 --model_name microsoft/Phi-3.5-vision-instruct --benchmark_name vqa2`

Run all the benchmarks for all the models across different quantization modes. This will lead to the creation of the `results.csv` file which will contain the detailed results

`python3 benchmark_launcher.py` 

### Video Inference

To perform video inference on a given video file and get the description of what is happening in the video run the `video_inference_demo.py`. We can change the model, quantization mode and video path

Run the following command : 

`python3 video_inference_demo.py --quantization_mode 8 --model_name microsoft/Phi-3.5-vision-instruct --video_path demo_video/sample_demo_1.mp4` s

## Pruned Models

1. Run the corresponding file in the `phi3_pruning_code/` for which we want to FT and compute the benchmark results for.
2. For eg. If we want to prune the Phi3.5 vision model and then FT on the Train split of VQAv2, we will run the command `python3 phi_prune_vqa2.py`

## Low-Rank Models

- The module [low_rank.py](./low_rank/low_rank.py) contains the code for following:
    - Replace `nn.Linear` layers with `LowRankLinear` layers. A `LowRankLinear` layer consists of two consecutive low-rank `nn.Linear` layers which approximate the original `nn.Linear`.
    - This module also records the metadata of the linear layers being replaced. This could be stored in a JSON file, so that we could save and re-load the low-rank model whenever required.
- The module [qwen2_low_rank.py](./low_rank/qwen2_low_rank.py) contains code for doing low rank factorization of QWen2-VL-2B Instruct model and finetuning it on Flickr30k dataset. The modules [qwen2_low_rank_vqa2.py](./low_rank/qwen2_low_rank_vqa2.py) and [qwen2_low_rank_docvqa.py](./low_rank/qwen2_low_rank_docvqa.py) contains the code for finetuning the low-rank model on VQA2 and DocVQA benchmarks respectively.
    - These modules also support pruning of Qwen2-VL on the corresponding datasets. The parameter passed to the main function determines if we'll perform low-rank factorization for a specific retained variance ration or if we would do the pruning (as introduced in the previous section for Phi-3.5 models).


## Plotting

Once the benchmark_launcher.py has finished we can plot the graphs by running `python3 plot.py` 