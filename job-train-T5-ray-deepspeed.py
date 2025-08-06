"""
Refactored script to fine-tune a T5 model with DeepSpeed ZeRO-3,
Ray Train, and Ray Data, including dynamic Ray cluster setup.
"""
import subprocess
import os
import time
import datetime
import sys
from tempfile import TemporaryDirectory
import cml.workers_v1 as workers
import torch
import deepspeed
from deepspeed.accelerator import get_accelerator
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

import ray
import ray.train
from ray.train import Checkpoint, DataConfig, ScalingConfig
from ray.train.torch import TorchTrainer

class StreamLogger:
    """
    Redirects stdout/stderr to both a file and the original console.
    This captures all print statements and logs from subprocesses.
    """
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log_file = open(log_file_path, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
        
    def __getattr__(self, attr):
        return getattr(self.terminal, attr)


def train_func(config):
    """Your training function that will be launched on each worker."""

    # Unpack training configs
    set_seed(config["seed"])
    model_id = config["model_id"]
    num_epochs = config["num_epochs"]
    train_batch_size = config["train_batch_size"]
    generation_max_length = config["generation_max_length"]
    deepspeed_config = config["deepspeed_config"]

    # Instantiate the Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare Ray Data Loaders
    train_ds = ray.train.get_dataset_shard("train")

    def collate_fn(batch):
        """Prepares a batch for the T5 model."""
        instructions = [str(s) for s in batch["instruction"]]
        inputs_col = [str(s) for s in batch["input"]]
        targets = [str(s) for s in batch["output"]]
        inputs = [f"Instruction: {instr}\nInput: {inp}" for instr, inp in zip(instructions, inputs_col)]
        
        model_inputs = tokenizer(
            inputs, max_length=256, padding="longest", truncation=True, return_tensors="pt"
        )
        labels = tokenizer(
            text_target=targets, max_length=generation_max_length, padding="longest", truncation=True, return_tensors="pt"
        ).input_ids
        
        labels[labels == tokenizer.pad_token_id] = -100
        model_inputs["labels"] = labels
        return model_inputs

    train_dataloader = train_ds.iter_torch_batches(
        batch_size=train_batch_size, collate_fn=collate_fn
    )
    # ====================================================

    # Initialize DeepSpeed Engine
    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=deepspeed_config,
    )
    device = get_accelerator().device_name(model.local_rank)

    # --- Training Loop ---
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            model.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
        
        # Report checkpoint and metrics to Ray Train
        # ==============================================================
        metrics = {"loss": loss.item(), "epoch": epoch}
        with TemporaryDirectory() as tmpdir:
            model.save_checkpoint(tmpdir)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            ray.train.report(
                metrics=metrics, checkpoint=Checkpoint.from_directory(tmpdir)
            )
        # ==============================================================
        
        if model.global_rank == 0:
            print(f"Epoch {epoch} finished. Reported metrics: {metrics}")


if __name__ == "__main__":
    sys.stdout = StreamLogger("raytrain.log")
    sys.stderr = sys.stdout # Capture errors as well

    # --- Define Number of Workers ---
    NUM_WORKERS = 1

    # --- Start Ray Head Node ---
    DASHBOARD_IP = os.environ['CDSW_IP_ADDRESS']
    ray_head_log = open("ray_head.log", "w")
    command = "ray start --head --block --include-dashboard=true --dashboard-port=$CDSW_READONLY_PORT --num-cpus=0 --num-gpus=0 &" 
    subprocess.run(
        command, shell=True, executable="/bin/bash", stdout=ray_head_log, stderr=subprocess.STDOUT
    )

    with open("RAY_HEAD_IP", 'w') as output_file:
        output_file.write(DASHBOARD_IP)
                
    ray_head_addr = DASHBOARD_IP + ':6379'
    worker_start_cmd = f"!ray start --block --address={ray_head_addr}"
    time.sleep(7)

    # --- Launch Ray Worker Nodes ---
    ray_workers = workers.launch_workers(
        n=NUM_WORKERS, cpu=2, memory=112, nvidia_gpu=1, code=worker_start_cmd,
    )
    time.sleep(15)

    deepspeed_config = {
        "optimizer": {"type": "AdamW", "params": {"lr": 2e-5}},
        "scheduler": {"type": "WarmupLR", "params": {"warmup_num_steps": 100}},
        "fp16": {"enabled": False},
        "bf16": {"enabled": True}, #If your GPU support it to prevent gradient overflow in fp16.
        "zero_optimization": {
            "stage": 3,
            "offload_optimizer": {"device": "none"},
            "offload_param": {"device": "none"},
        },
        "gradient_accumulation_steps": 1,
        "gradient_clipping": True,
        "steps_per_print": 10,
        "train_micro_batch_size_per_gpu": 8,
        "wall_clock_breakdown": False,
    }

    training_config = {
        "seed": 42,
        "model_id": "/home/cdsw/t5-3b",
        "num_epochs": 1,
        "train_batch_size": 8,
        "generation_max_length": 128,
        "deepspeed_config": deepspeed_config,
    }
    
    # Use only the first 20 rows from the Parquet file for training
    train_dataset = ray.data.read_parquet("wikisql/data/train-00000-of-00001-36d5d5ed0289390f.parquet").limit(1000)
    
    ray_datasets = {"train": train_dataset}
    trainer = TorchTrainer(
        train_func,
        train_loop_config=training_config,
        scaling_config=ScalingConfig(
            num_workers=NUM_WORKERS, # Use the variable here
            use_gpu=True
        ),
        datasets=ray_datasets,
        dataset_config=DataConfig(datasets_to_split=["train"]),
    )
    
    start_time = time.time()
    result = trainer.fit()
    end_time = time.time()
    elapsed_seconds = end_time - start_time
    time_str = str(datetime.timedelta(seconds=int(elapsed_seconds)))

    # --- Results ---
    print("\n" + "="*50)
    print("Training finished!")
    print(f"Total training time: {time_str}")
    print(f"Final loss: {result.metrics['loss']:.4f}")
    
    best_checkpoint_path = result.best_checkpoints[0][0].path
    print(f"Best checkpoint saved at: {best_checkpoint_path}")
    print("="*50 + "\n")