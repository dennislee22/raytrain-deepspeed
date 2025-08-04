# DeepSpeed with Ray
<img width="406" height="179" alt="image" src="https://github.com/user-attachments/assets/a03094fc-7087-4246-936c-e9ba7db84255" />

## Platform Requirement
☑️ Python 3.11

☑️ Cloudera AI(CAI)/Cloudera Machine Learning (CML) 1.5.x

## Finetune the T5-3B model using Ray Train with Deepspeed
- Objective: The script fine-tunes the large t5-large model for a text-to-SQL task using the wikisql dataset.
- Distributed Training: It leverages Ray to orchestrate a multi-worker training cluster, distributing the data and computation across multiple nodes, each with its own GPU. 🚀
- Memory Optimization: It uses DeepSpeed's ZeRO Stage 3 to partition the model's parameters, gradients, and optimizer states across the VRAM of all three GPUs in the Ray cluster.
- Overcoming Limitations: By sharding the model components, DeepSpeed drastically reduces the memory required on any single GPU, enabling the fine-tuning of a model that would be too large to fit into one GPU's limited RAM.

### Procedure
1. Create a new CAI project.
   
2. Install python libraries.
  ```
  pip install accelerate torch transformers ipywidgets pandas numpy ray[default]
  ```

3. Download the LLM into the project of the CAI/CML platform using either `git clone` or `wget`.
Example:
  ```
  git lfs clone https://huggingface.co/google-t5/t5-large
  ```

```
$ ls -lh ./t5-large/
total 11G
-rw-r--r--. 1 cdsw cdsw 1.2K Jul 30 01:11 config.json
-rw-r--r--. 1 cdsw cdsw 2.8G Jul 30 01:51 flax_model.msgpack
-rw-r--r--. 1 cdsw cdsw  147 Jul 30 01:11 generation_config.json
-rw-r--r--. 1 cdsw cdsw 2.8G Jul 30 02:18 model.safetensors
-rw-r--r--. 1 cdsw cdsw 2.8G Jul 30 02:12 pytorch_model.bin
-rw-r--r--. 1 cdsw cdsw 8.3K Jul 30 01:11 README.md
-rw-r--r--. 1 cdsw cdsw 774K Jul 30 01:11 spiece.model
-rw-r--r--. 1 cdsw cdsw 2.8G Jul 30 02:04 tf_model.h5
-rw-r--r--. 1 cdsw cdsw 1.4M Jul 30 01:11 tokenizer.json
```

4. Download the [wikisql](https://huggingface.co/datasets/htriedman/wikisql) dataset. See the sample content of the dataset in parquet format.
Example:
- Dataset:
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
cdsw@7002fw5g8zs6rqgo:~$ python view-parquet.py wikisql/data/test-00000-of-00001-a07b94a320bf63eb.parquet 

Reading data from 'wikisql/data/test-00000-of-00001-a07b94a320bf63eb.parquet'...

Contents of the Parquet file:
                                    instruction  ...                                             output
0      Translate the following into a SQL query  ...  SELECT Nationality FROM table WHERE Player = T...
1      Translate the following into a SQL query  ...  SELECT School/Club Team FROM table WHERE Years...
2      Translate the following into a SQL query  ...  SELECT School/Club Team FROM table WHERE Years...
3      Translate the following into a SQL query  ...  SELECT COUNT School/Club Team FROM table WHERE...
4      Translate the following into a SQL query  ...      SELECT Round FROM table WHERE Circuit = Assen
...                                         ...  ...                                                ...
15873  Translate the following into a SQL query  ...  SELECT Points FROM table WHERE Year > 1972 AND...
15874  Translate the following into a SQL query  ...        SELECT Chassis FROM table WHERE Points = 39
15875  Translate the following into a SQL query  ...  SELECT Points FROM table WHERE Engine = ford v...
15876  Translate the following into a SQL query  ...  SELECT Chassis FROM table WHERE Engine = ford ...
15877  Translate the following into a SQL query  ...  SELECT SUM Year FROM table WHERE Entrant = elf...

[15878 rows x 3 columns]

Schema of the Parquet file:
instruction: string
input: string
output: string
-- schema metadata --
huggingface: '{"info": {"features": {"instruction": {"dtype": "string", "' + 116
```

🗒️ Test 1: Train `T5-11B model` with single GPU


🗒️ Test 2: Train `T5-11B model` with 1 Ray worker of 1 GPU
<img width="900" height="663" alt="image" src="https://github.com/user-attachments/assets/1bce3ccd-306f-4ad0-8523-8134f500a878" />

🗒️ Test 3: Train `T5-11B model` with 3 Ray workers of 1 GPU each.
<img width="1400" height="709" alt="image" src="https://github.com/user-attachments/assets/b41a6049-0eea-47e3-9c74-0465ad08ecc0" />

```
NAME               READY   STATUS    RESTARTS   AGE   IP             NODE                                          NOMINATED NODE   READINESS GATES
4dvzodfufpiyqnvu   5/5     Running   0          12m   10.42.9.204    ares-ecs-ws-gpu01.ares.olympus.cloudera.com   <none>           <none>
iz04ymk241xhby7y   5/5     Running   0          12m   10.42.9.205    ares-ecs-ws-gpu01.ares.olympus.cloudera.com   <none>           <none>
poei8fuvp2f68yj6   5/5     Running   0          12m   10.42.8.61     ares-ecs-ws06.ares.olympus.cloudera.com       <none>           <none>
qlaw1c8tm2ksqsp7   5/5     Running   0          12m   10.42.11.234   ares-ecs-ws-gpu03.ares.olympus.cloudera.com   <none>           <none>
```
- Because of ZeRO-3, the model is sharded across 3 workers. `Ray Train` is designed to handle the complexities of gathering the sharded model state and saving a single, consolidated checkpoint.
- ScalingConfig: The num_workers is set to 3 to ensure the job utilizes all available GPU workers.
  

<img width="1449" height="355" alt="image" src="https://github.com/user-attachments/assets/584bdb46-9f39-46c3-8345-4ab23ca8a6c2" />



## Tips

⚠️  Ray's object store is configured to use only 42.9% of available memory (13.8GB out of 32.3GB total). For optimal Ray Data performance, we recommend setting the object store to at least 50% of available memory. You can do this by setting the 'object_store_memory' parameter when calling ray.init() or by setting the RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION environment variable.
