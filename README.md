# bigbird-example

- pip install -r requirements.txt
- set the batch_size to the desired size <= 64 in github_example.py
- python github_example.py

This will init the model, run a forward and back call and sleep for 20 seconds so that the memory usage can be seen in nvidia-smi.

With 40GB of memory I can just about get a forwards call with a batch size of 4
