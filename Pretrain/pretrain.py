import torch
# from Korpora import Korpora
from transformers import PegasusConfig, PegasusTokenizer, PegasusForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import json
import os
from transformers import BertTokenizer
from torch.utils.data import Dataset
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"사용 중인 디바이스: {device}")

train_dataset = torch.load("/home/suyeon1803/PEGASUS_a100/data/final_pt/nonpre_conv_kowiki.pt")

print("토큰화 결과 로드 완료")
print(f"train_dataset의 크기: {len(train_dataset)}")

# 입력 데이터를 텐서로 변환 (CPU에서 처리)
input_ids = torch.tensor([item['input_ids'] for item in train_dataset]).cpu()
attention_mask = torch.tensor([item['attention_mask'] for item in train_dataset]).cpu()
labels = torch.tensor([item['labels'] for item in train_dataset]).cpu()

# CustomDataset 클래스 정의
class CustomDataset(Dataset):
    def __init__(self, input_ids, attention_mask, labels): # 클래스의 생성자 메소드 
        # 클래스의 인스턴스 생성할 때 자동으로 호출 / 인스턴스 생성 시 초기화 작업 수행 
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# 데이터셋 생성 (CPU 텐서 사용)
train_dataset_tensors = CustomDataset(input_ids, attention_mask, labels)

# Pegasus 모델 구성 설정
config = PegasusConfig(
    vocab_size=96000,
    max_position_embeddings=1024,
    encoder_layers=12,
    decoder_layers=12,
    encoder_attention_heads=16,
    decoder_attention_heads=16,
    d_model=1024,
)
tokenizer_path = '/home/suyeon1803/PEGASUS_a100/PEGASUS_tokenizer2'

# 모델 및 토크나이저 로드 (GPU로 이동)
tokenizer = PegasusTokenizer(
    vocab_file=f"{tokenizer_path}/tokenizer.model",
    special_tokens_map_file=f"{tokenizer_path}/special_tokens_map.json",  
    tokenizer_config_file=f"{tokenizer_path}/tokenizer_config.json"  
)
model = PegasusForConditionalGeneration(config).to(device) # 언어 모델링 헤드가 있어 요약에 사용 가능 
model.resize_token_embeddings(len(tokenizer))

print("모델 로드 완료")

data_collator = DataCollatorForSeq2Seq( # 배치 형성하는 객체 
    tokenizer=tokenizer,
    model=model,
    padding='longest',
    max_length=256,
)

# TrainingArguments 설정
training_args = TrainingArguments( 
    output_dir="/home/suyeon1803/PEGASUS_a100/new_pretrain_checkpoing",
    learning_rate=3e-5,
    per_device_train_batch_size=32, # 학습 시간, gpu 사용량 확인해 조정 
    gradient_accumulation_steps=4,
    num_train_epochs=100,
    save_steps=30_000,
    save_total_limit=5,
    fp16=True,  # GPU 사용 시 성능 향상
)

# Trainer 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset_tensors,
    data_collator=data_collator,
)

print("학습 준비 완료")

# 학습 시작
trainer.train()

# 모델과 토크나이저 저장
model.save_pretrained("/home/suyeon1803/PEGASUS_a100/new_pretrain_model")
tokenizer.save_pretrained("/home/suyeon1803/PEGASUS_a100/new_pretrain_model")