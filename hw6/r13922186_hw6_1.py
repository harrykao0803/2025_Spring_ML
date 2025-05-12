from transformers import (
    AutoModelForCausalLM, # imports the model for causal language modeling
    AutoTokenizer, # imports the tokenizer for the model
    BitsAndBytesConfig, # imports the configuration for using bitsandbytes
    pipeline # imports the pipeline for text generation
)
from peft import (
    LoraConfig, # imports the configuration for LoRA
    get_peft_model, # imports the function to get the PEFT model
    PeftModel # imports the PEFT model
)
import os
import json
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = '0' # Sets the CUDA device to use
device = torch.device('cuda:0') # Creates a CUDA device object
from datasets import Dataset # Imports the Dataset class from the datasets library
from trl import SFTConfig, SFTTrainer # Imports the SFTConfig and SFTTrainer classes from the trl library
import random
random.seed(42) # Sets the random seed for reproducibility
from tqdm import tqdm # Imports the tqdm library for progress bars
import csv


### Load Model & Tokenizer
sft_model_name = 'unsloth/Llama-3.2-1B-Instruct' # Specifies the name of the pre-trained model to use
sft_bnb_config = BitsAndBytesConfig( # Configuration for using bitsandbytes
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
sft_model = AutoModelForCausalLM.from_pretrained( # Loads the pre-trained model
    pretrained_model_name_or_path=sft_model_name,
    quantization_config=sft_bnb_config,
    low_cpu_mem_usage=True,
)
sft_tokenizer = AutoTokenizer.from_pretrained( # Loads the tokenizer for the model
    pretrained_model_name_or_path=sft_model_name,
)
sft_tokenizer.add_special_tokens({'pad_token': '[PAD]'}) # Adds a special token for padding
peft_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.2, # TODO: Add dropout (Strong Baseline - 3)
    bias='none',
    task_type='CAUSAL_LM',
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)
peft_model = get_peft_model(sft_model, peft_config)


### Dataset Formatting Functions
def load_jsonlines(file_name: str):
    f = open(file_name, 'r')
    return [json.loads(line) for line in f]


def nshot_chats(nshot_data: list, n: int, question: str, answer: any, mode: str) -> dict: # Function to create n-shot chats
    if mode not in ['train', 'test']:
        raise AssertionError('Undefined Mode!!!')

    chats = []
    
    # TODO: Use fixed few-shot examples (Strong Baseline - 1)
    # Sort examples by answer length (descending)
    sorted_qna = sorted(nshot_data, key=lambda x: len(x['answer']), reverse=True)
    
    if len(sorted_qna) < n:
        raise ValueError(f"Not enough examples to select {n} few-shots.")

    selected_qnas = sorted_qna[100:100+n]
    
    for qna in selected_qnas:  # for qna in random.sample(nshot_data, n): # Randomly sample n examples from the n-shot data
        chats.append(
            {
                'role': 'user',
                'content': f'Q: {qna["question"]}' # Creates a user message with the question
            }
        )
        chats.append(
            {
                'role': 'assistant',
                'content': f'A: {qna["answer"]}' # Creates an assistant message with the answer
            }
        )

    chats.append(
        {
            'role': 'user',
            'content': f'Q: {question} Let\'s think step by step. At the end, you MUST write the answer as an integer after \'####\'.' # Creates a user message with the question and instructions
        }
    )
    if mode == 'train':
        chats.append(
            {
                'role': 'assistant',
                'content': f'A: {answer}' # Creates an assistant message with the answer
            }
        )

    return chats # Returns the list of chats


### Format GSM8K Data for Fine-tuning
gsm8k_train = load_jsonlines('./dataset/gsm8k_train_self-instruct.jsonl') # You can use refined gsm8k_train_self-instruct.jsonl for fine-tuning


# gsm8k_train = sorted(gsm8k_train, key=lambda x: len(x['answer']), reverse=True)
# for i, qna in enumerate(gsm8k_train): # Iterates over the GSM8K training data
#     if i == 200:
#         break
#     print(qna)
#     print(len(str(qna['answer'])))
#     print("=" * 80)


formatted_gsm8k = []
TRAIN_N_SHOT = 8 # TODO: Give model more examples (Medium Baseline - 3)
max_token_len = 0 # Record token length of dataset and prevent data from truncation
for qna in gsm8k_train: # Iterates over the GSM8K training data
    chats = nshot_chats(nshot_data=gsm8k_train, n=TRAIN_N_SHOT, question=qna['question'], answer=qna['answer'], mode='train') # Creates n-shot chats for the current example
    train_sample = sft_tokenizer.apply_chat_template(chats, tokenize=False) # Applies the chat template to the chats
    train_sample = train_sample[train_sample.index("<|eot_id|>") + len("<|eot_id|>"):] # Remove Cutting Knowledge Date in prompt template
    formatted_gsm8k.append( # Appends the formatted example to the list
        {
            'text': train_sample # Adds the text of the example
        }
    )
    max_token_len = max(max_token_len, len(sft_tokenizer(train_sample)['input_ids'])) # Updates the maximum token length

formatted_gsm8k = Dataset.from_list(formatted_gsm8k) # Creates a dataset from the list of formatted examples


### Fine-tuning
# trainer
training_arguments = SFTConfig( # Configuration for the SFT trainer
    seed=1126,
    data_seed=1126,
    output_dir=f"./sft",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_32bit",
    num_train_epochs=5, # TODO: If you use fixed few-shot examples, increase epoch (Strong Baseline - 5)
    logging_strategy="steps",
    logging_steps=0.1,
    save_strategy="steps",
    save_steps=0.1,
    lr_scheduler_type='linear',
    learning_rate=1e-4, # TODO: Decrease learning rate (Medium Baseline - 2)
    weight_decay=1e-2, # TODO: Add weight decay (Strong Baseline - 2)
    bf16=True,
    group_by_length=True,
    max_seq_length=max_token_len,
    dataset_text_field='text',
    report_to='none',
)
trainer = SFTTrainer( # Creates the SFT trainer
    model=peft_model,
    train_dataset=formatted_gsm8k,
    peft_config=peft_config,
    processing_class=sft_tokenizer,
    args=training_arguments,
)
trainer.train() # Starts the training process