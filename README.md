# Ray Train with DeepSpeed 🚀

<img width="711" height="333" alt="image" src="https://github.com/user-attachments/assets/fb499809-e797-4d8c-9580-584b735fc49d" />

## Finetune a huge model using Ray Train with Deepspeed
- Objective: The script [job-train-T5-ray-deepspeed.py](job-train-T5-ray-deepspeed.py) fine-tunes the large T5-3b/T5-11b model for a text-to-SQL task using the wikisql dataset.
- Distributed Training: It leverages Ray to orchestrate a multi-worker training cluster, distributing the data and computation across multiple nodes, each with its own GPU. 🚀
- Memory Optimization: It uses DeepSpeed's ZeRO Stage 3 to partition the model's parameters, gradients, and optimizer states across the GRAM of all three GPUs in the Ray cluster.
- Overcoming Limitations: By sharding the model components, DeepSpeed drastically reduces the memory required on any single GPU, enabling the fine-tuning of a model that would be too large to fit into one GPU's limited RAM.

## Platform Requirement
☑️ Python 3.11/3.10

☑️ Cloudera AI(CAI)/Cloudera Machine Learning (CML) 1.5.x

### Procedure
1. Create a new CAI project.
   
2. Install python libraries.
  ```
  pip install accelerate torch transformers ipywidgets pandas numpy ray[default]
  ```

3. Download the LLM into the project of the CAI/CML platform using either `git clone` or `wget`.
Example:
  ```
  git lfs clone https://huggingface.co/google-t5/t5-3b
  git lfs clone https://huggingface.co/google-t5/t5-11b
  ```

```
$ ls -lh ./t5-3b/
total 32G
-rw-r--r-- 1 cdsw cdsw 1.2K Aug  4 01:22 config.json
-rw-r--r-- 1 cdsw cdsw  11G Aug  4 02:19 model.safetensors
-rw-r--r-- 1 cdsw cdsw  11G Aug  4 02:23 pytorch_model.bin
-rw-r--r-- 1 cdsw cdsw 7.7K Aug  4 01:22 README.md
-rw-r--r-- 1 cdsw cdsw 774K Aug  4 01:22 spiece.model
-rw-r--r-- 1 cdsw cdsw  11G Aug  4 02:24 tf_model.h5
-rw-r--r-- 1 cdsw cdsw 1.4M Aug  4 01:22 tokenizer.json
```

4. Download the [wikisql](https://huggingface.co/datasets/htriedman/wikisql) dataset. See the sample content of the dataset in parquet format.
Dataset example:
```
$ python view-parquet.py wikisql/data/train-00000-of-00001-36d5d5ed0289390f.parquet 

Reading data from 'wikisql/data/train-00000-of-00001-36d5d5ed0289390f.parquet'...

Contents of the Parquet file:
                                    instruction  ...                                             output
0      Translate the following into a SQL query  ...  SELECT Notes FROM table WHERE Current slogan =...
1      Translate the following into a SQL query  ...  SELECT Current series FROM table WHERE Notes =...
2      Translate the following into a SQL query  ...  SELECT Format FROM table WHERE State/territory...
3      Translate the following into a SQL query  ...  SELECT Text/background colour FROM table WHERE...
4      Translate the following into a SQL query  ...  SELECT COUNT Fleet Series (Quantity) FROM tabl...
...                                         ...  ...                                                ...
56350  Translate the following into a SQL query  ...           SELECT Time FROM table WHERE Score = 3-2
56351  Translate the following into a SQL query  ...  SELECT Ground FROM table WHERE Opponent = asto...
56352  Translate the following into a SQL query  ...  SELECT Competition FROM table WHERE Ground = s...
56353  Translate the following into a SQL query  ...  SELECT COUNT Decile FROM table WHERE Name = re...
56354  Translate the following into a SQL query  ...   SELECT Report FROM table WHERE Circuit = tripoli

[56355 rows x 3 columns]

Schema of the Parquet file:
instruction: string
input: string
output: string
-- schema metadata --
huggingface: '{"info": {"features": {"instruction": {"dtype": "string", "' + 116
```

🗒️ Test 1: Train `T5-11B model` with single GPU

When fine-tuning a large model like `T5-11B`, GRAM is quickly consumed by three main components:
- Model Weights: The parameters of the model itself. A model with billions of parameters requires gigabytes of GRAM just to be loaded.
- Gradients: During backpropagation, the network calculates a gradient for every parameter. These gradients are roughly the same size as the model weights.
- Optimizer States: Modern optimizers like Adam or AdamW store additional data for each parameter (e.g., momentum and variance), often doubling the memory required by the model weights.

For example, a 11-billion parameter model using standard 32-bit precision would require:
- Model Weights: 11B params×4 bytes/param≈44 GB
- Gradients: ≈44 GB
- Optimizer States (Adam): ≈88 GB
Total: ≈176 GB

This total far exceeds the capacity of even high-end GPUs like the A100 (80 GB). As a result, train `T5-11B model` with single GPU hits the following error.
```
torch.OutOfMemoryError: CUDA out of memory
```

🗒️ Test 2: Train `T5-11B model` with 3 Ray workers of 1 GPU each.
Thanks to Ray Train for orchestrating the distributed backend and DeepSpeed for its ZeRO-3 optimization, we successfully fine-tuned a model far too large for a single machine's memory. Ray scales a training workload, allowing DeepSpeed to effectively partition the model's parameters, gradients, and optimizer states across the entire cluster. By leveraging this powerful combination, we transformed a memory-bound failure on a single node into a successful, distributed training job, aggregating the GPU memory from all workers to fit the huge model.
The script uses `bf16` to save memory.

<img width="900" height="696" alt="image" src="https://github.com/user-attachments/assets/8f99d0b9-5c91-47ff-a702-75b2627f86d7" />

```
NAME               READY   STATUS    RESTARTS   AGE   IP             NODE                                          NOMINATED NODE   READINESS GATES
4dvzodfufpiyqnvu   5/5     Running   0          12m   10.42.9.204    ares-ecs-ws-gpu01.ares.olympus.cloudera.com   <none>           <none>
iz04ymk241xhby7y   5/5     Running   0          12m   10.42.9.205    ares-ecs-ws-gpu01.ares.olympus.cloudera.com   <none>           <none>
poei8fuvp2f68yj6   5/5     Running   0          12m   10.42.8.61     ares-ecs-ws06.ares.olympus.cloudera.com       <none>           <none>
qlaw1c8tm2ksqsp7   5/5     Running   0          12m   10.42.11.234   ares-ecs-ws-gpu03.ares.olympus.cloudera.com   <none>           <none>
```

🗒️ Test 3: Train `T5-3B model` with single GPU

<img width="900" height="636" alt="image" src="https://github.com/user-attachments/assets/b4cbd395-1cc3-4f3c-be94-d1fafe777f7e" />

🗒️ Test 4: Train `T5-3B model` with 2 Ray workers of 1 GPU each.

<img width="900" height="699" alt="image" src="https://github.com/user-attachments/assets/1e2c1f1e-64f4-486a-ab2d-562eadc7de17" />

🗒️ Test 5: Train `T5-3B model` with 3 Ray workers of 1 GPU each.

<img width="900" height="701" alt="image" src="https://github.com/user-attachments/assets/87486c54-e6ae-493c-8b80-440b656f6235" />

### Result Overview

<img width="860" height="205" alt="image" src="https://github.com/user-attachments/assets/6cf3c063-3c11-4067-a1f8-df19c8315448" />

Training/Finetuning the same model using 1 worker/GPU seems faster than using 2 or 3 workers because in a distributed environment with multiple pods/nodes, it involves sending data over the network. Due to this overhead of network communication, training model in a distributed environment might take longer than using 1 worker/GPU.
  
### How the results are collected?
- Because of ZeRO-3, the model is sharded across 3 workers. `Ray Train` is designed to handle the complexities of gathering the sharded model state and saving a single, consolidated checkpoint.
- When using DeepSpeed's ZeRO Stage 3, the model's weights, gradients, and optimizer states are partitioned across all the GPUs/workers. No single worker has the complete model. So, `model.save_checkpoint(tmpdir)` tells each worker to save only its own part of the model (its "shard") into a temporary, local directory on that specific worker's machine.
- `torch.distributed.barrier()` is a synchronization point. It forces every worker to pause and wait at this line until all other workers have also reached it. This ensures that no worker moves on to the next step until every other worker has successfully saved its local shard.
- Each worker calls `ray.train.report()` to Ray Train controller. The Ray Train controller receives these reports from all workers. When it sees that each worker has reported a checkpoint shard, it performs the final assembly:
   - It creates a new, permanent checkpoint directory in the main experiment folder (e.g., ~/ray_results/TorchTrainer_.../checkpoint_000001/).
   - It then copies the checkpoint shards from each worker's temporary directory over the network and places them into that single, final checkpoint directory.
This way, Ray Train abstracts away the complexity of distributed storage. The workers only need to worry about saving their local piece, and the central controller handles the aggregation, resulting in a complete, unified checkpoint that can be used for resuming training or for inference.
  



