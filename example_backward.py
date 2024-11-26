
import torch
from transformers import AutoTokenizer
from lxt.models.llama import LlamaForCausalLM, attnlrp
import matplotlib.pyplot as plt
import numpy as np
import os

device = "cuda:0"
model_path = "./Llama-2-7b-chat-hf"
if not os.path.exists(model_path):
    model_path = "./Llama-2-7b-hf"

def save_heatmap(values, tokens, figsize, title, save_path):
    fig, ax = plt.subplots(figsize=figsize)  # Increase the size of the figure

    abs_max = abs(values).max()
    im = ax.imshow(values, cmap='bwr', vmin=-abs_max, vmax=abs_max)
    
    layers = np.arange(values.shape[-1])

    ax.set_xticks(np.arange(len(layers)))
    ax.set_yticks(np.arange(len(tokens)))

    ax.set_xticklabels(layers)
    ax.set_yticklabels(tokens)

    plt.title(title)
    plt.xlabel('Layers')
    plt.ylabel('Tokens')
    plt.colorbar(im)

    plt.show()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')

def hidden_relevance_hook(module, input, output):
    if isinstance(output, tuple):
        output = output[0]
    module.hidden_relevance = output.detach().cpu()


if __name__ == "__main__":

    # load model & apply AttnLRP
    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map=device, force_download=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    attnlrp.register(model)

    # apply hooks
    for layer in model.model.layers:
        layer.register_full_backward_hook(hidden_relevance_hook)

    # forward & backard pass
    prompt_response = f"<s>I have 5 cats and 3 dogs. How many cats do I have again? Ah, I have "
    input_ids = tokenizer(prompt_response, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    input_embeds = model.get_input_embeddings()(input_ids)

    output_logits = model(inputs_embeds=input_embeds.requires_grad_(), use_cache=False).logits
    max_logits, max_indices = torch.max(output_logits[:, -1, :], dim=-1)
    max_logits.backward(max_logits)

    print("Prediction:", tokenizer.convert_ids_to_tokens(max_indices))

    # trace relevance through layers
    relevance_trace = []
    for i in range(0, 32):
        # TODO: double ceck this --
        relevance = model.model.layers[i].attention.hidden_relevance[0].sum(-1)
        # ----
        relevance = relevance / relevance.abs().max()
        relevance_trace.append(relevance)

    relevance_trace = torch.stack(relevance_trace)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    save_heatmap(relevance_trace.float().numpy().T, tokens, (20, 10), f"Latent Relevance Trace (Normalized)", f'latent_rel_trace.png')

# syntax vs. semantics
# -> predicting next token in a sequence
# activation patching
# one sentence regarding syntax, one regarding semantics

# 2 tasks, -> check relevances for each, then compare the differences in activations, across layers
# how does in-context change things?
# polysemantic -> superposition ; anthropic -> sparse autoencoder, unpack vectors that are not orthonormal -> Zs of the autoencoder turn out to be monosemantic
# extracting interpretable features from calude 3 sonnet
# ->


