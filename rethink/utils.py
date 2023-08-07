from torch import softmax
from torch.distributions import Categorical
import torch
from tqdm.notebook import tqdm 
from transformers import AutoModelForCausalLM, AutoTokenizer


def ends_with_word(ids, word, tokenizer):
    return word == tokenizer.decode(ids, skip_special_tokens=True)[-len(word):]

def next_token_distr(model, ids):
    logits = model(ids.unsqueeze(0))['logits'][0][-1]
    probs = softmax(logits, dim=0)
    return probs


def generate_next_token(model, ids, return_distr=False):
    probs = next_token_distr(model, ids)
    sample = Categorical(probs).sample()
    return sample if not return_distr else sample, probs


@torch.no_grad()
def generate_tokens(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, ids, stop_word, max_length=50, return_distr=False):
    i = 0
    answer = []
    next_token = -1

    bar = tqdm(total=max_length)

    if return_distr:
        distr = []
    
    while i < max_length and next_token != tokenizer.eos_token_id and not ends_with_word(ids, stop_word, tokenizer):
        next_token = generate_next_token(model, ids, return_distr=return_distr)
          
        if return_distr:
            next_token, next_distr = next_token
            distr.append(next_distr)

        ids = torch.cat((ids,next_token.unsqueeze(0)), dim=0)
        answer.append(next_token)
        bar.update()
        i += 1
    while i < max_length:
        bar.update()
        i += 1

    if return_distr:    
        distr = torch.stack(distr)
    answer = torch.stack(answer)

    return answer if not return_distr else answer, distr


