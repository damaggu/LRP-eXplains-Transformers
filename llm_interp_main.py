# -*- coding: utf-8 -*-
"""llm_interp (JP version).ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1J1UsOUqZHjKjeCoaFvvGdN_5x-ydTbze

First we have to install Layer-Wise relevance propagation https://pypi.org/project/lxt/
"""


# reset GPU memory
from numba import cuda

device = cuda.get_current_device()
device.reset()

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM
from lxt.models.llama import LlamaForCausalLM
import lxt.models.llama as lxt_llama
from lxt.models.bert import BertForMaskedLM
import lxt.models.bert as lxt_bert
from lxt.utils import pdf_heatmap, clean_tokens, heatmap
import lxt.functional as lf


class Gauge:
    def __init__(self, model_path, track_attention=False):
        self.model_path = model_path
        self.accelerator = Accelerator()
        if 'llama' in model_path.lower():
            self.model = lxt_llama.LlamaForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16,
                                                                    output_attentions=track_attention)
            if self.model_path == "meta-llama/Llama-2-7b-hf":
                self.model.gradient_checkpointing_enable()
            lxt_llama.attnlrp.register(self.model)
        elif 'bert' in model_path.lower():
            self.model = lxt_bert.BertForMaskedLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16,
                                                                  output_attentions=track_attention)
            lxt_bert.attnlrp.register(self.model)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, torch_dtype=torch.bfloat16,
                                                                output_attentions=track_attention)
            # attnlrp
        self.model = self.accelerator.prepare(self.model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path)  # AutoTokenizer is an abstract factory that creates an instance of the pecific tokenizer for TinyLlama
        # apply Attention-aware LRP (AttnLRP) rules

    def compute_relevance(self, prompt, target="logits", track_attention=False):
        input_ids = self.tokenizer(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(self.model.device)
        input_embeds = self.model.get_input_embeddings()(input_ids)

        output = self.model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False, output_attentions=True)
        output_logits = output.logits

        if target == "logits":
            output = output_logits
        elif target == "probas":
            output = lf.softmax(output_logits, -1)

        max_logits, max_indices = torch.max(output[0, -1, :], dim=-1)
        max_logits.backward(max_logits)

        relevance = input_embeds.grad.float().sum(-1).cpu()[0]

        # normalize relevance between [-1, 1] for plotting
        relevance = relevance / relevance.abs().max()

        # remove '_' characters from token strings
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
        tokens = clean_tokens(tokens)

        if track_attention:
            attention = output[-1]
            return tokens, relevance, attention
        return tokens, relevance


gauge_tinyL = Gauge("TinyLlama/TinyLlama-1.1B-Chat-v1.0", track_attention=True)

prompt = """\
Context: Mount Everest attracts many climbers, including highly experienced mountaineers. There are two main climbing routes, one approaching the summit from the southeast in Nepal (known as the standard route) and the other from the north in Tibet. While not posing substantial technical climbing challenges on the standard route, Everest presents dangers such as altitude sickness, weather, and wind, as well as hazards from avalanches and the Khumbu Icefall. As of November 2022, 310 people have died on Everest. Over 200 bodies remain on the mountain and have not been removed due to the dangerous conditions. The first recorded efforts to reach Everest's summit were made by British mountaineers. As Nepal did not allow foreigners to enter the country at the time, the British made several attempts on the north ridge route from the Tibetan side. After the first reconnaissance expedition by the British in 1921 reached 7,000 m (22,970 ft) on the North Col, the 1922 expedition pushed the north ridge route up to 8,320 m (27,300 ft), marking the first time a human had climbed above 8,000 m (26,247 ft). The 1924 expedition resulted in one of the greatest mysteries on Everest to this day: George Mallory and Andrew Irvine made a final summit attempt on 8 June but never returned, sparking debate as to whether they were the first to reach the top. Tenzing Norgay and Edmund Hillary made the first documented ascent of Everest in 1953, using the southeast ridge route. Norgay had reached 8,595 m (28,199 ft) the previous year as a member of the 1952 Swiss expedition. The Chinese mountaineering team of Wang Fuzhou, Gonpo, and Qu Yinhua made the first reported ascent of the peak from the north ridge on 25 May 1960. \
Question: How high did they climb in 1922? According to the text, the 1922 expedition reached 8,"""

tokens, relevance, attention = gauge_tinyL.compute_relevance(prompt, "logits", track_attention=True)

print(relevance.shape)

# from bertviz import model_view
# model_view(attention, tokens)

heatmap(tokens,relevance)