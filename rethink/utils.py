from torch import softmax
from torch.distributions import Categorical
import torch
from tqdm.notebook import tqdm 
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate_next_token(model, ids):
    logits = model(ids)['logits'][-1]
    probs = softmax(logits, dim=0)
    return Categorical(probs).sample()


def generate_tokens(model: AutoModelForCausalLM, tokenizer: AutoTokenizer, ids, stop_ids, max_length):
    i = 0
    answer = []
    next_token = -1

    bar = tqdm(total=max_length)
    with torch.no_grad():
        while i < max_length and next_token != tokenizer.eos_token_id and not torch.all(torch.eq(ids[-len(stop_ids):], stop_ids)):
            next_token = generate_next_token(model, ids)
            ids = torch.cat((ids,next_token.unsqueeze(0)), dim=0)
            answer.append(next_token)
            bar.update()
            i += 1
        while i < max_length:
            bar.update()
            i += 1

    return answer
