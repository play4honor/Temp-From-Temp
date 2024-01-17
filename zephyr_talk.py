import torch
from transformers import pipeline
from transformers.generation.logits_process import LogitsProcessorList

from warp import GPUTemperatureLogitsWarper

pipe = pipeline(
    "text-generation",
    model="HuggingFaceH4/zephyr-7b-beta",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# We use the tokenizer's chat template to format each message - see https://huggingface.co/docs/transformers/main/en/chat_templating
messages = [
    {
        "role": "system",
        "content": "You are a friendly chatbot",
    },
    {
        "role": "user",
        "content": "What would be a good name for a project that modifies an LLM's generation temperature according to changes in the actual GPU temperature?",
    },
]
prompt = pipe.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
outputs = pipe(
    prompt,
    max_new_tokens=256,
    do_sample=True,
    # temperature=0.7,
    top_p=0.95,
    logits_processor=LogitsProcessorList([GPUTemperatureLogitsWarper()]),
)
print(outputs[0]["generated_text"].split("<|assistant|>\n")[1])
