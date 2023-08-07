from transformers import AutoModelForCausalLM, AutoTokenizer
from rethink.utils import generate_tokens, next_token_distr
import torch

default_prompt = "{}\nH: {}\nM: "
default_stop = "H:"


def process_context(
    request,
    context,
    rethinking_prompt,
    tokenizer,
    log=False,
):
    rethinking_request = rethinking_prompt.format(
        context,
        request
    )
    ids = tokenizer.encode(rethinking_request, return_tensors='pt', add_special_tokens=False)[0]
    if log:
        print(f'=== request:\n{rethinking_request}\n')
    ids = torch.cat([torch.tensor([tokenizer.bos_token_id]), ids], dim=0)
    return ids


@torch.no_grad()
def answer(
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        request,
        context,
        rethinking_prompt=default_prompt, 
        max_length=50,
        stop_word=default_stop,
        log=False,
    ):
    # generate prompt for context-aware generation
    ids = process_context(request, context, rethinking_prompt, tokenizer, log).to(model.device)
    
    answer, distr = generate_tokens(model, tokenizer, ids, stop_word, max_length)

    answer_str = tokenizer.decode(answer, skip_special_tokens=True)

    return {
        'ans_ids': answer,
        'distr': distr,
        'answer': answer_str
    }

@torch.no_grad()
def distr_for_answer(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    request,
    context,
    ans_ids,
    rethinking_prompt=default_prompt,
    log=False,
):  
    ids = torch.cat([
        process_context(request, context, rethinking_prompt, tokenizer, log).to(model.device),
        ans_ids.to(model.device)
    ], dim=0)
    ans_len = len(ans_ids)
    logits = model(ids.unsqueeze(0))['logits'][0][-ans_len-1:-1]

    distrs = torch.softmax(logits, dim=-1)
    return distrs


def rethink_context(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    request,
    context,
    prior_context,
    rethinking_prompt=default_prompt, 
    max_length=50,
    stop_word=default_stop,
    log=False,
):
    answer_dict = answer(
        model, 
        tokenizer, 
        request, 
        context, 
        rethinking_prompt, 
        max_length, 
        stop_word,
        log=log
    )

    ans_ids = answer_dict['ans_ids']
    distr = answer_dict['distr']

    prior_distr = distr_for_answer(
        model, 
        tokenizer,
        request, 
        prior_context,
        ans_ids,
        rethinking_prompt,
        log
    )

    dist = torch.abs(distr - prior_distr).sum(dim=-1) / 2

    tokens = tokenizer.batch_decode(ans_ids)
    ans = tokenizer.decode(ans_ids)
    return {
        'answer': ans,
        'tokens': tokens,
        'rethink': distr,
        'prior': prior_distr,
        'dist': dist
    }