import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load HF_TOKEN from environment variable
hf_token = os.getenv("HF_TOKEN")
# env:HF_TOKEN = "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"

tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilgpt2", token=hf_token)
text = "The Dark"
input_test = tokenizer(text, return_tensors="pt")
print(input_test)
output = model.generate(input_test['input_ids'], max_length=5, do_sample=False)
print(tokenizer.decode(output[0]))