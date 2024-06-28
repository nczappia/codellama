from datasets import load_dataset
from datasets import load_dataset
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import TrainingArguments, Trainer
import transformers
import torch
from trl import SFTTrainer
import os
import multiprocessing
from peft import LoraConfig
from transformers import (
    BitsAndBytesConfig,
    logging,
    set_seed,
)

# os.environ["CUDA_VISIBLE_DEVICES"]="0"


torch.cuda.empty_cache()

print(torch.cuda.is_available())
exit()

# Define your model ID, dataset details, and other training parameters
model_id = "bigcode/starcoder2-7b"
dataset_name = "iamtarun/code_instructions_120k_alpaca"
subset = "data/train-00000-of-00001-d9b93805488c263e.parquet"
dataset_text_field = "output"
split = "train"
max_seq_length = 512
max_steps = 1000
micro_batch_size = 1
gradient_accumulation_steps = 8
learning_rate = 2e-5
warmup_steps = 20

# Load the dataset using load_dataset
data = load_dataset(
        "iamtarun/code_instructions_120k_alpaca",
        data_files="data/train-00000-of-00001-d9b93805488c263e.parquet",
        split="train",
        token=os.environ.get("HF_TOKEN", None),
        num_proc= multiprocessing.cpu_count(),

)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)

tokenizer.pad_token = tokenizer.eos_token  # Set eos_token as the padding token


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

lora_config = LoraConfig(
    r=8,
    target_modules=[
        "q_proj",
        "o_proj",
        "k_proj",
        "v_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    task_type="CAUSAL_LM",
)



if torch.cuda.is_available():
    print("GPU")
    device = torch.device("cuda:0")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attention_dropout=0.1,
    )
else:
    #Handle no GPU availiability
    print("No GPU")
    device = torch.device("cpu")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attention_dropout=0.1,
    )

if torch.cuda.device_count() > 1:
    print("Using DataParallel for multiple GPUs")
    model = torch.nn.DataParallel(model)

model.to(device)



# Define training arguments for 2 GPUs
training_args = TrainingArguments(
    output_dir="./outputs",
    overwrite_output_dir=True,
    per_device_train_batch_size=micro_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    warmup_steps=warmup_steps,
    logging_steps=10,
    evaluation_strategy="steps",
    save_steps=1000,
    save_total_limit=2,  # Keep the most recent 2 checkpoints
    num_train_epochs=1,  # Adjust for multiple epochs if needed
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    fp16=True,  # Enable mixed precision for potentially faster training
    optim="paged_adamw_8bit",
    report_to="wandb",  # Optional: Report to Weights & Biases for logging (install wandb)
)


# Define a function to preprocess the data
"""
def preprocess_function(examples):
    return tokenizer(examples[dataset_text_field], 
                     truncation=True, padding="max_length", 
                     max_length=max_seq_length)

# Preprocess the training dataset
train_dataset = data.map(preprocess_function, batched=True)

#print("Model Summary:", torch.cuda.memory_summary())
"""




# Create a Trainer instance with 2 GPUs
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=data,
    #max_seq_length=512,
    #peft_config=lora_config,
    dataset_text_field="output",
)

# Start training!
trainer.train()
