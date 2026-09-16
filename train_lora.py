# train_lora.py
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from peft import LoraConfig, PeftModel
from trl import SFTTrainer

# ====================== 1. 配置参数 ======================
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
new_model = "mistral-lora-sft-math"

# 4bit量化配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# LoRA配置（你可以在这里做ablation：修改r，alpha）
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=3,
    logging_steps=10,
    save_strategy="epoch",
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
)

# ====================== 2. 构造简易数学问答数据集 ======================
# 注意：正式实验你可以扩充，这里只是最小样例
data_raw = [
    {"instruction": "求解一元二次方程 x^2 -5x +6=0", "output":"x=2 或 x=3"},
    {"instruction": "求导数 f(x)=x^3 + 2x", "output":"f'(x)=3x^2+2"},
    {"instruction": "解释什么是矩阵的秩", "output":"矩阵秩是矩阵线性无关行/列向量的最大数目"}
]
dataset = Dataset.from_list(data_raw)

# 格式化prompt模板，Mistral对话模板
def format_prompt(sample):
    return f"<s>[INST] {sample['instruction']} [/INST] {sample['output']} </s>"

def formatting_func(example):
    output_texts = []
    for i in range(len(example['instruction'])):
        text = format_prompt({"instruction": example["instruction"][i], "output": example["output"][i]})
        output_texts.append(text)
    return output_texts

# ====================== 3. 加载模型与tokenizer ======================
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
model.config.use_cache = False
model.config.pretraining_tp = 1

# ====================== 4. SFT Trainer训练 ======================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    formatting_func=formatting_func,
    max_seq_length=512,
    tokenizer=tokenizer,
    args=training_args,
)

# 开始训练
trainer.train()
trainer.save_model(new_model)

# ====================== 5. 推理测试 ======================
pipe = pipeline(
    "text-generation",
    model=new_model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
prompt = "<s>[INST] 求矩阵 [[1,2],[3,4]] 的行列式 [/INST]"
outputs = pipe(prompt, max_new_tokens=128, temperature=0.7)
print(outputs[0]["generated_text"])