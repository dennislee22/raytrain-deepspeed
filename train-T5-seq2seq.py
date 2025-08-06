import torch
import datetime
import time
import pandas as pd
import sys
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    set_seed,
)

class StreamLogger:
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        # Open the log file in write mode, which will create it if it doesn't exist
        # or overwrite it if it does.
        self.log_file = open(log_file_path, "w")

    def write(self, message):
        # Write the message to the original console (terminal)
        self.terminal.write(message)
        # Write the same message to the log file
        self.log_file.write(message)
        # Flush both streams to ensure output is written immediately
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def __getattr__(self, attr):
        # This is a fallback to ensure that if any other attribute of stdout
        # is accessed, it's retrieved from the original terminal object.
        return getattr(self.terminal, attr)

class ParquetSqlDataset(Dataset):
    def __init__(self, file_path, tokenizer, num_samples=None, input_max_length=128, target_max_length=128):
        # Note: You will need to have 'pandas' and 'pyarrow' installed.
        # pip install pandas pyarrow
        print(f"Reading Parquet file from: {file_path}")
        self.df = pd.read_parquet(file_path)
        
        # Use a subset of the data if num_samples is specified
        if num_samples:
            self.df = self.df.head(num_samples)
            print(f"Using the first {num_samples} samples.")
        else:
            print("Using the entire dataset.")

        self.tokenizer = tokenizer
        self.input_max_length = input_max_length
        self.target_max_length = target_max_length
        self.prefix = "translate English to SQL: "

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieves and processes a single data sample from the dataset.
        This method is called by the DataLoader for each item.
        """
        # Get the row from the DataFrame
        row = self.df.iloc[idx]
        
        # Prepare the input and target texts
        question = self.prefix + row['instruction'] 
        
        # FIX: Changed `row['sql']['human_readable']` to `row['output']` to match the
        # likely schema of the user's data, based on the provided traceback.
        target = row['output']

        # Tokenize the input text
        model_inputs = self.tokenizer(
            question,
            max_length=self.input_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tokenize the target text to create the labels
        labels = self.tokenizer(
            text_target=target,
            max_length=self.target_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids # We only need the input_ids for the labels

        # The tokenizer returns tensors with a batch dimension, so we squeeze it out
        input_ids = model_inputs['input_ids'].squeeze()
        attention_mask = model_inputs['attention_mask'].squeeze()
        labels = labels.squeeze()

        # Replace the pad_token_id in labels with -100 to be ignored in loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def main():
    set_seed(42)
    model_id = "/home/cdsw/t5-11b"
    repository_id = f"output"
    
    learning_rate = 2e-5
    num_train_epochs = 1
    train_batch_size = 8
    generation_max_length = 128
    
    print(f"Loading tokenizer and model for '{model_id}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    print("Loading and preparing dataset from local Parquet file...")
    local_parquet_path = "wikisql/data/train-00000-of-00001-36d5d5ed0289390f.parquet"

    # Instantiate the custom dataset. By removing the 'num_samples' argument,
    # the ParquetSqlDataset class will use the entire dataset by default.
    train_dataset = ParquetSqlDataset(
        file_path=local_parquet_path,
        tokenizer=tokenizer,
        target_max_length=generation_max_length
    )
    
    print(f"Training on {len(train_dataset)} examples.")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=-100
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=repository_id,
        per_device_train_batch_size=train_batch_size,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        logging_dir=f"{repository_id}/logs",
        logging_strategy="steps",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=2,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    print("\n" + "="*50)
    print("Starting training...")
    start_time = time.time()
    
    trainer.train()
    
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    time_str = str(datetime.timedelta(seconds=int(elapsed_seconds)))
    print("Training finished!")
    print(f"Total training time: {time_str}")
    print("="*50 + "\n")

    # --- 5. Save Model and Tokenizer ---
    print(f"Saving model and tokenizer to '{repository_id}'")
    trainer.save_model(repository_id)
    tokenizer.save_pretrained(repository_id)
    
    print("\nFine-tuning complete. Model saved.")


if __name__ == "__main__":
    log_file_path = "finetune_log.txt"
    sys.stdout = StreamLogger(log_file_path)
    # Redirect stderr to the same logger to capture any error messages
    sys.stderr = sys.stdout

    print(f"--- Starting Fine-Tuning Script ---")
    print(f"Log file will be saved to: {log_file_path}")
    main()
    print(f"--- Script Finished ---")
