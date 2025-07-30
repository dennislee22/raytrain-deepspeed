# Deepspeed with Ray

Python 3.11

## Tips

⚠️  Ray's object store is configured to use only 42.9% of available memory (13.8GB out of 32.3GB total). For optimal Ray Data performance, we recommend setting the object store to at least 50% of available memory. You can do this by setting the 'object_store_memory' parameter when calling ray.init() or by setting the RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION environment variable.


- Model:
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

- Test 1: 1 worker with 1 GPU
<img width="900" height="663" alt="image" src="https://github.com/user-attachments/assets/1bce3ccd-306f-4ad0-8523-8134f500a878" />

- Test 2: 3 workers with 1 GPU each.
<img width="900" height="706" alt="image" src="https://github.com/user-attachments/assets/88a1c001-2fed-4147-b96e-f8f9338ca247" />


