import re
import json
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from src.model.core.load_model import predict


MODEL_NAME = "Qwen/Qwen3-4B"
def load_llm(model_name=MODEL_NAME):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        cache_dir="cache",
        dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,)
    return model

def load_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./cache", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def create_pipeline():
    model = load_llm()
    tokenizer = load_tokenizer(MODEL_NAME)
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=300,
        temperature=0.3,
        top_p=0.8,
    )
    return pipe

tokens = load_tokenizer(MODEL_NAME)
pipe = create_pipeline()

def clean_output(text):
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

def generate_text(user_input,system_prompt):
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

    prompt = tokens.apply_chat_template(messages,tokenize= False,add_generation_prompt=True)
    answer = pipe(prompt)[0]['generated_text']
    output = answer.replace(prompt, "").strip()

    return clean_output(output)





