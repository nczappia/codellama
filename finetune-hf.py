from datasets import load_dataset
from datasets import load_dataset
import os
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer



# Define your model ID, dataset details, and other training parameters
model_id = "bigcode/starcoder2-7b"
dataset_name = "iamtarun/code_instructions_120k_alpaca"
subset = "data/train-00000-of-00001-d9b93805488c263e.parquet"
dataset_text_field = "output"
split = "train"
max_seq_length = 1024
max_steps = 10000
micro_batch_size = 8
gradient_accumulation_steps = 8
learning_rate = 2e-5
warmup_steps = 20

# Load the dataset using load_dataset
data = load_dataset(
        "iamtarun/code_instructions_120k_alpaca",
        data_files="data/train-00000-of-00001-d9b93805488c263e.parquet",
        split="train",
        token=os.environ.get("HF_TOKEN", None),
)

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

# Define training arguments for 2 GPUs
training_args = TrainingArguments(
    output_dir="./outputs",
    overwrite_output_dir=True,
    per_device_train_batch_size=micro_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    warmup_steps=warmup_steps,
    logging_steps=100,
    evaluation_strategy="steps",
    save_steps=1000,
    save_total_limit=2,  # Keep the most recent 2 checkpoints
    num_train_epochs=1,  # Adjust for multiple epochs if needed
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    logging_dir="./logs",
    fp16=True,  # Enable mixed precision for potentially faster training
    report_to="wandb",  # Optional: Report to Weights & Biases for logging (install wandb)
)

# Define a function to preprocess the data
def preprocess_function(examples):
    return tokenizer(examples[dataset_text_field], truncation=True, padding="max_length")

# Preprocess the training dataset
train_dataset = data.map(preprocess_function, batched=True)

# Create a Trainer instance with 2 GPUs
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Start training!
trainer.train()
