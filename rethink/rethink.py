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
        return_distr=False,
        log=False,
    ):
    # generate prompt for context-aware generation
    ids = process_context(request, context, rethinking_prompt, tokenizer, log).to(model.device)
    
    new_tokens = generate_tokens(model, tokenizer, ids, stop_word, max_length, return_distr)
    if return_distr:
        answer, distr = new_tokens

    answer = tokenizer.decode(answer, skip_special_tokens=True)

    # if answer[-len(stop_word):] == stop_word:
        # answer = answer[:-len(stop_word)]
    return answer if not return_distr else answer, distr


@torch.no_grad()
def distr_for_answer(
    model: AutoModelForCausalLM, 
    tokenizer: AutoTokenizer, 
    request,
    context,
    answer,
    rethinking_prompt=default_prompt,
    log=False,
):
    ids = process_context(request, context, rethinking_prompt, tokenizer, log).to(model.device)
    ans_ids = tokenizer.encode(answer, add_special_tokens=False)

    distrs = []
    for ans_tok in ans_ids:
        distr = next_token_distr(model, ids)
        distrs.append(distr.unsqueeze(0))


        ids = torch.cat([
            ids,
            torch.tensor(ans_tok, device=model.device).unsqueeze(0)
        ], dim=0)
    
    distrs = torch.cat(distrs)

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
    ans, distr = answer(
        model, 
        tokenizer, 
        request, 
        context, 
        rethinking_prompt, 
        max_length, 
        stop_word, 
        return_distr=True,
        log=log
    )
    prior_distr = distr_for_answer(
        model, 
        tokenizer,
        request, 
        prior_context,
        ans,
        rethinking_prompt,
        log
    )

    dist = torch.abs(distr - prior_distr).sum(dim=-1) / 2

    ids = tokenizer(ans, add_special_tokens=False)['input_ids']
    tokens = tokenizer.batch_decode(ids, skip_special_tokens=True)
    return {
        'answer': ans,
        'tokens': tokens,
        'rethink': distr,
        'prior': distr,
        'dist': dist
    }