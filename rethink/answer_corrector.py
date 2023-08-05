from transformers import AutoModelForCausalLM, AutoTokenizer
from rethink.utils import generate_tokens
import torch


default_prompt = "{}\nH: {}\nM: "
default_stop = "H:"


def process_context(
    request,
    context,
    rethinking_prompt,
    tokenizer
):
    rethinking_request = rethinking_prompt.format(
        context,
        request
    )
    ids = tokenizer.encode(rethinking_request, return_tensors='pt')[0]
    return ids


def correct_answer(
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        request,
        context,
        rethinking_prompt=default_prompt, 
        max_length=50,
        stop_word=default_stop,
        return_distr=False
    ):
    # generate prompt for context-aware generation
    ids = process_context(request, context, rethinking_prompt, tokenizer).to(model.device)
    stop_ids = tokenizer.encode(stop_word, return_tensors='pt')[0].to(model.device)
    
    new_tokens = generate_tokens(model, tokenizer, ids, stop_ids, max_length, return_distr)
    if return_distr:
        answer, distr = new_tokens

    if torch.all(answer[-len(stop_ids):] == stop_ids):

        answer = answer[:-len(stop_ids)]
    answer = tokenizer.decode(answer, skip_special_tokens=True)
    return answer if not return_distr else answer, distr
