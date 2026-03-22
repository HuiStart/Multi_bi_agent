"""
Fine-tuning utilities for multi-bi-agent training
"""
import os
import json
import torch
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from torch.utils.data import DataLoader, Dataset
from accelerate import Accelerator
from accelerate import FullyShardedDataParallelPlugin, gradient_accumulation_split
from tqdm import tqdm


class CustomDataset(Dataset):
    """Custom dataset for fine-tuning"""
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_text = item['input']
        output_text = item['output']

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )

        labels = self.tokenizer(
            output_text,
            return_tensors="pt",
            max_length=self.max_length,
            padding="max_length",
            truncation=True
        )

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels['input_ids'].squeeze(0)
        }


def load_model_and_tokenizer(model_name, use_lora=False, lora_r=8, lora_alpha=16, lora_dropout=0.1):
    """
    Load model and tokenizer with optional LoRA configuration
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    if use_lora:
        print("Configuring LoRA...")
        peft_config = LoraConfig(
            task_type=TaskType.CAU,  # Causal Language Modeling
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Adjust based on model architecture
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    return model, tokenizer


def prepare_data(data_file):
    """
    Load and prepare training data
    """
    print(f"Loading data from {data_file}")

    if data_file.endswith('.json'):
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

    elif data_file.endswith('.jsonl'):
        data = []
        with open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))

    return data


def train_model(
    model,
    tokenizer,
    train_data,
    output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    max_length=512,
    gradient_accumulation_steps=4,
    eval_strategy="no",
    logging_steps=50
):
    """
    Train the model using the provided data
    """
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        deepspeed="ds_config.json" if os.path.exists("ds_config.json") else None
    )

    dataset = CustomDataset(train_data, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=per_device_train_batch_size, shuffle=True)

    print(f"Number of training samples: {len(dataset)}")
    print(f"Number of batches: {len(dataloader)}")

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        warmup_steps=100,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy=eval_strategy,
        logging_steps=logging_steps,
        save_strategy="epoch",
        load_best_model_at_end=False,
        report_to="tensorboard"
    )

    trainer = Trainer(
        model=accelerator.prepare(model),
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True),
        tokenizer=tokenizer
    )

    print("Starting training...")
    trainer.train()

    print(f"Training completed. Model saved to {output_dir}")
    return trainer


def fine_tune_lora(model, tokenizer, train_data, output_dir, r=8, alpha=16, dropout=0.1):
    """
    Fine-tune with LoRA configuration
    """
    from peft import LoraConfig, get_peft_model, TaskType

    peft_config = LoraConfig(
        task_type=TaskType.CAU,
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    return train_model(model, tokenizer, train_data, output_dir)


def test_model(model, tokenizer, test_data, max_length=256):
    """
    Test the model on provided test data
    """
    from transformers import GenerationConfig

    print("Testing model...")
    model.eval()

    results = []
    for item in tqdm(test_data, desc="Testing"):
        input_text = item['input']
        inputs = tokenizer(input_text, return_tensors="pt", max_length=max_length, padding=True, truncation=True)
        input_ids = inputs['input_ids'].to(model.device)
        attention_mask = inputs['attention_mask'].to(model.device)

        generation_config = GenerationConfig(
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1
        )

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config
        )

        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        results.append({
            'input': input_text,
            'output': output_text
        })

    return results


def main():
    """
    Example usage of fine-tuning utilities
    """
    model_name = "gpt2"  # or any other model name
    data_file = "train_data.json"  # Replace with your data file
    output_dir = "fine_tuned_model"

    # Load data
    train_data = prepare_data(data_file)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_name, use_lora=True)

    # Train model
    train_model(model, tokenizer, train_data, output_dir)

    print("Fine-tuning complete!")


if __name__ == "__main__":
    main()
