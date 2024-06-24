"""
Adapted from: https://github.com/Vision-CAIR/MiniGPT-4/blob/main/demo.py
"""
import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from video_llama.common.config import Config
from video_llama.common.dist_utils import get_rank
from video_llama.common.registry import registry
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation,SeparatorStyle,conv_llava_llama_2
import decord
decord.bridge.set_bridge('torch')

#%%
# imports modules for registration
from video_llama.datasets.builders import *
from video_llama.models import *
from video_llama.processors import *
from video_llama.runners import *
from video_llama.tasks import *

from generate_idk import *

#%%
def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--model_type", type=str, default='vicuna', help="The type of LLM")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))
model.eval()
vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')

# ========================================
#             Gradio Setting
# ========================================


if args.model_type == 'vicuna':
    chat_state = default_conversation.copy()
else:
    chat_state = conv_llava_llama_2.copy()
    chat_state.system =  "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
img_list = []

gr_video="/home/ubuntu/op_1_0320241830.mp4"
user_message="Describe what you see in the video"

llm_message = chat.upload_video_without_audio(gr_video, chat_state, img_list)
chat.ask(user_message, chat_state)
output_text, _, _, u_wordlist, wordlist, p_list, p_all = chat.answer_lure(conv=chat_state,
                                                                    img_list=img_list,
                                                                    num_beams=1,
                                                                    temperature=1,
                                                                    max_new_tokens=300,
                                                                    max_length=2000)
print(f"""
\n
========================================
Original (possibly hallucinatory) output
========================================
\n
{output_text}
""")                                                                    

output_text = replace_words_with_idk(output_text, wordlist, p_all, un=0.9)
print(f"""
\n
=================================================
Replacing possible hallucination words with [IDK]
=================================================
\n
{output_text}
""")
rewrite_prompt = 'According to the video, remove the information that does not exist in the following description: ' + output_text
chat_state.append_message(chat_state.roles[0], "<Video><ImageHere></Video> "+ rewrite_prompt)
output_text, _, _, _, _, _, _ = chat.answer_lure(conv=chat_state,
                                            img_list=img_list,
                                            num_beams=1,
                                            temperature=1,
                                            max_new_tokens=300,
                                            max_length=2000)
print(f"""
\n
============
Final Output
============
\n
{output_text}
""")
