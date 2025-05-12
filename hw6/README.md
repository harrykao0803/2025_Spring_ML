# 2025 Spring ML HW6

r13922186 高永杰

## PC Configuraiton

- **OS**: Ubuntu 22.04
- **GPU**: RTX 3090
- **CUDA**: 11.7
- **Python**: 3.10.12

## File Overview
```txt
hw6/
├── r13922186_hw6_1.py
├── r13922186_hw6_2.py
├── hw6_intro.pdf
└── README.md
```

- `r13922186_hw6_1.py` : finetuning code
- `r13922186_hw6_2.py` : inferencing code
- `hw6_intro.pdf` : introduction to this homework

## Environment Building

- Open terminal

- Build virtual environment
  ```sh
  conda create -n ml_hw6 python=3.10.12
  conda activate ml_hw6
  ```

- Install required packages
  ```sh
  pip install -U datasets trl bitsandbytes
  pip install peft
  ```

## Download Dataset

- Create a `dataset/` folder inside `r13922186_hw6/`

- Change directory to `r13922186_hw6/dataset/`

- Run the following commands
  ```sh
  wget https://www.csie.ntu.edu.tw/~b10902031/gsm8k_train.jsonl
  wget https://www.csie.ntu.edu.tw/~b10902031/gsm8k_train_self-instruct.jsonl
  wget https://www.csie.ntu.edu.tw/~b10902031/gsm8k_test_public.jsonl
  wget https://www.csie.ntu.edu.tw/~b10902031/gsm8k_test_private.jsonl
  wget https://www.csie.ntu.edu.tw/~b10902031/ailuminate_test.csv
  ```
  
- The downloaded datasets will be stored in `r13922186_hw6/dataset/`

## How to run

### 1. Finetune

- Change directory to `r13922186_hw6/`
  
- Run the following command
  ```sh
  python r13922186_hw6_1.py
  ```

- The resulted checkpoints will be stored in `r13922186_hw6/sft/`

### 2. Inference

- Create an `output/` folder inside `r13922186_hw6/`

- Change directory to `r13922186_hw6/`
  
- Run the following command
  ```sh
  python r13922186_hw6_2.py
  ```

- The resulted .txt file `r13922186.txt` will be stored in `r13922186_hw6/output/`


## References

- Some codes are advised by ChatGPT
  - line 65~72 of `r13922186_hw6_1.py`, I told ChatGPT my idea (e.g. choose the ones with longer answers) and asked him to generate the code.
    ```python
    # TODO: Use fixed few-shot examples (Strong Baseline - 1)
    # Sort examples by answer length (descending)
    sorted_qna = sorted(nshot_data, key=lambda x: len(x['answer']), reverse=True)
    
    if len(sorted_qna) < n:
        raise ValueError(f"Not enough examples to select {n} few-shots.")

    selected_qnas = sorted_qna[100:100+n]
    ```

  - line 103~105 of `r13922186_hw6_2.py`, I ask ChatGPT how to implement greedy decoding strategy. <br>
    ChatGPT link: https://chatgpt.com/share/681dc788-b600-8005-adc5-6c60d261f984
    
    ```python
    # TODO: Use greedy decoding strategy (Medium Baseline - 5)
    num_beams=1,
    do_sample=False,
    ```

