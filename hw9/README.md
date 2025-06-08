# 2025 Spring ML HW9

r13922186 高永杰

## PC Configuraiton

- **OS**: Ubuntu 22.04
- **GPU**: RTX 3090
- **CUDA**: 11.7
- **Python**: 3.10.12

## File Overview
```txt
r13922186_hw9/
├── hw9_intro.pdf
├── peft-ml2025-hw9.zip
├── r13922186_hw9_1.ipynb
└── README.md
```

- `hw9_intro.pdf`: introduction to this homework
- `peft-ml2025-hw9.zip`: compressed file of my own PEFT package
- `r13922186_hw9_1.ipynb`: main model merging code

## Peft

- Add a merging method (i.e. sce) into the original PEFT package

- Files modified: 
  - `peft-ml2025-hw9\src\peft\tuners\lora\model.py`
  - `peft-ml2025-hw9\src\peft\utils\merge_utils.py`

## Environment Building

- Open terminal

- Build virtual environment
  ```sh
  conda create -n ml_hw9 python=3.10.12
  conda activate ml_hw9
  ```

## How to run

- Unzip `peft-ml2025-hw9.zip` 

- Open `r13922186_hw9_1.ipynb`

- Simply click `Run All`

