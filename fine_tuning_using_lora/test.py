from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model="./fine_tuned_model",
    tokenizer="./fine_tuned_model",
)

result = pipe(
    "### Human: tell me about vuyyalawada kiran education details?\n### Assistant:",
    max_new_tokens=50
)

print(result[0]["generated_text"])