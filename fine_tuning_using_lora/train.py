from transformers import Trainer, TrainingArguments, AutoModelForCausalLM ,AutoTokenizer
from peft import LoraConfig
from datasets import load_dataset
from trl import SFTTrainer
model_name= "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model= AutoModelForCausalLM.from_pretrained(model_name)
tokenizer= AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
dataset = load_dataset("json", data_files="data.json")
train_dataset = dataset["train"]
peft=LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    task_type="CAUSAL_LM",
)
trianing_args= TrainingArguments(
    output_dir="./fine_tuned_model",
    per_device_train_batch_size=1,
    num_train_epochs=8,
    logging_steps=1)

trainer=SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    peft_config=peft,
    processing_class=tokenizer,
    args=trianing_args
)

trainer.train()
trainer.save_model()
tokenizer.save_pretrained("./fine_tuned_model")