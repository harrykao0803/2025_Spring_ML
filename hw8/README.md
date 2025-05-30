# 2025 Spring ML HW8

r13922186 高永杰


## Environment
- Platform: Kaggle
- Accelerator: GPU T4 × 2
- Language: Python


## File Overview
```txt
hw8/
├── r13922186_hw8_1.ipynb
└── README.md
```

- `r13922186_hw8_1.ipynb` : model editing code, includes both ROME and MEMIT methods


## How to Run
1. Open Kaggle website
2. Import `r13922186_hw8_1.ipynb` into your notebook
3. Select specified accelerator and language
4. Find these lines in the code and choose the model editing method you want
   ```sh
   ###### TODO: Change the method :) ######
   # RewritingParamsClass, apply_method, hparam = FTHyperParams, apply_ft_to_model, ft_hparam
   # RewritingParamsClass, apply_method, hparam = ROMEHyperParams, apply_rome_to_model, rome_hparam
   RewritingParamsClass, apply_method, hparam = MEMITHyperParams, apply_memit_to_model, memit_hparam
   ```
5. Click the "Save Version" button on the top right corner
6. The code is now running ^^


## References
- ROME: https://github.com/kmeng01/rome
- MEMIT: https://github.com/kmeng01/memit

