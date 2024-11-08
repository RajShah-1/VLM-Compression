import torch
from mplug_owl_video.modeling_mplug_owl import MplugOwlForConditionalGeneration
from transformers import AutoTokenizer
from mplug_owl_video.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from transformers import BitsAndBytesConfig

# pretrained_ckpt = '/home/hice1/drauthan3/scratch/models--MAGAer13--mplug-owl-llama-7b-video'  # Adjust this path
pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b-video'
model = MplugOwlForConditionalGeneration.from_pretrained(
    pretrained_ckpt,
    torch_dtype=torch.bfloat16,
    load_in_4bit=True
)
model.to("cuda:0")  # Adjust the device if needed

image_processor = MplugOwlImageProcessor.from_pretrained(pretrained_ckpt)
tokenizer = AutoTokenizer.from_pretrained(pretrained_ckpt)
processor = MplugOwlProcessor(image_processor, tokenizer)

prompts = [
'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
Human: <|video|>
Human: What is the woman doing in the video?
AI: ''']

# generate_kwargs = {
#     'do_sample': False,
#     'top_k': 5,
#     'max_length': 48,
#     'temperature': 0.5,
#     'top_p': 0.9,
#     'num_beams': 1,
#     'no_repeat_ngram_size': 2,
#     'early_stopping': True,
#     'length_penalty': 1
# }
generate_kwargs = {
    'do_sample': True,
    'top_k': 5,
    'max_length': 512
}

# def getDescription(prompts, video_list, generate_kwargs, nframes):
#     inputs = processor(text=prompts, videos=video_list, num_frames=nframes, return_tensors='pt')
#     inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
#     inputs = {k: v.to(model.device) for k, v in inputs.items()}
#     with torch.no_grad():
#         res = model.generate(**inputs, **generate_kwargs)
#     sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
#     return sentence

video_path = "/home/hice1/drauthan3/scratch/MSRVTT/videos/all/video1.mp4"
video_list = [video_path]
# print("Printing description...")
# print(getDescription(prompts, video_list, generate_kwargs, 48))

inputs = processor(text=prompts, videos=video_list, num_frames=4, return_tensors='pt')
inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
inputs = {k: v.to(model.device) for k, v in inputs.items()}
with torch.no_grad():
    res = model.generate(**inputs, **generate_kwargs)
sentence = tokenizer.decode(res.tolist()[0], skip_special_tokens=True)
print(sentence)
print(model.get_memory_footprint())