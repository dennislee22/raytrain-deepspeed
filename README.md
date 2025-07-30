# Deepspeed with Ray

Python 3.11

## Tips

⚠️  Ray's object store is configured to use only 42.9% of available memory (13.8GB out of 32.3GB total). For optimal Ray Data performance, we recommend setting the object store to at least 50% of available memory. You can do this by setting the 'object_store_memory' parameter when calling ray.init() or by setting the RAY_DEFAULT_OBJECT_STORE_MEMORY_PROPORTION environment variable.

⚠️  ValueError: The configured object store size (15.032385536 GB) exceeds /dev/shm size (4.500000768 GB). This will harm performance. Consider deleting files in /dev/shm or increasing its size with --shm-size in Docker. To ignore this warning, set RAY_OBJECT_STORE_ALLOW_SLOW_STORAGE=1.

<img width="900" height="663" alt="image" src="https://github.com/user-attachments/assets/1bce3ccd-306f-4ad0-8523-8134f500a878" />
