# 2025 Spring ML HW10

r13922186 高永杰

## PC Configuraiton

- **OS**: Ubuntu 22.04
- **GPU**: RTX 3090
- **CUDA**: 11.7
- **Python**: 3.10.12

## File Overview
```txt
r13922186_hw10/
├── r13922186_hw10_1.ipynb
├── r13922186_hw10_2.ipynb
├── r13922186_hw10_3.ipynb
├── r13922186_hw10_4.ipynb
├── r13922186_hw10_5.ipynb
├── r13922186_hw10_6.ipynb
└── README.md
```

- `r13922186_hw10_1.ipynb` : code for object-1
- `r13922186_hw10_2.ipynb` : code for object-2
- `r13922186_hw10_3.ipynb` : code for object-3
- `r13922186_hw10_4.ipynb` : code for object-4
- `r13922186_hw10_5.ipynb` : code for object-5
- `r13922186_hw10_6.ipynb` : code for object-6

## Environment Building

- Open terminal

- Build virtual environment
  ```sh
  conda create -n ml_hw10 python=3.10.12
  conda activate ml_hw10
  ```

- Install required packages
  ```sh
  pip install ipykernel==6.29.5
  ```

## How to run

- Change directory to `/r13922186_hw10`
  ```sh
  cd /your/path/to/r13922186_hw10
  ```

- Open VScode
  ```sh
  code .
  ```

- There are **6 notebooks** to be executed **in order**:
  1. `r13922186_hw10_1.ipynb`
  2. `r13922186_hw10_2.ipynb`
  3. `r13922186_hw10_3.ipynb`
  4. `r13922186_hw10_4.ipynb`
  5. `r13922186_hw10_5.ipynb`
  6. `r13922186_hw10_6.ipynb`
   
- Open each notebook and click the **"Run All"** button from the top menu one by one.

## Result

- For each object, 15 images are generated and saved in the `/results` folder

- `results.zip` is the compressed file of folder `/results`

- If an object is trained using Custom Diffusion, the generated checkpoints will be saved in the `/checkpoints` folder