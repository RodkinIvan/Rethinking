from transformers import AutoModelForCausalLM, AutoTokenizer
from rethink.utils import generate_tokens

default_prompt = "H: {}\nM: {}\nH: {}\nH: {}\nM: "
default_stop = "H:"


def correct_answer(
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer, 
        request, 
        answer, 
        response, 
        rethinking_prompt=default_prompt, 
        max_length=50,
        stop_word=default_stop,
    ):
    rethinking_request = rethinking_prompt.format(
        request, 
        answer, 
        response, 
        request
    )
    ids = tokenizer.encode(rethinking_request, return_tensors='pt')[0].to(model.device)
    stop_ids = tokenizer.encode(stop_word, return_tensors='pt')[0].to(model.device)
    answer = generate_tokens(model, tokenizer, ids, stop_ids, max_length)
    return tokenizer.decode(answer, skip_special_tokens=True)