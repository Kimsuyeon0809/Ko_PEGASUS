import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration, Trainer, TrainingArguments
import json
import os
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

# GPU가 사용 가능한지 확인
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# JSON 파일을 순회하며 utterances, summaries 추출
def load_data_from_json(json_dir):
    utterances = []
    summaries = []
    for filename in os.listdir(json_dir):
        if filename.endswith('.json'):
            with open(os.path.join(json_dir, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)
                for dialogue_data in data['data']:
                    dialogue = dialogue_data['body']['dialogue']
                    summary = dialogue_data['body']['summary']

                    # 모든 발화를 하나의 텍스트로 결합 (사전학습된 모델의 입력으로 사용)
                    utterance_text = " ".join([utterance['utterance'] for utterance in dialogue])

                    utterances.append(utterance_text)
                    summaries.append(summary)
    return utterances, summaries

# Training 데이터 로드
train_json_dir = '/home/suyeon1803/PEGASUS_a100/한국어 대화 요약/Training/[라벨]한국어대화요약_train/[라벨]한국어대화요약_train' # 279992개
utterances_train, summaries_train = load_data_from_json(train_json_dir)

# Validation 및 Test 데이터 로드
valid_test_json_dir = '/home/suyeon1803/PEGASUS_a100/한국어 대화 요약/Validation/[라벨]한국어대화요약_valid/[라벨]한국어대화요약_valid' # 17502*2개
utterances_valid_test, summaries_valid_test = load_data_from_json(valid_test_json_dir)

utterances_valid, utterances_test, summaries_valid, summaries_test = train_test_split(
    utterances_valid_test, summaries_valid_test, test_size=0.5, random_state=42)

# Dataset 객체로 변환
train_dataset = Dataset.from_dict({
    'utterance': utterances_train,
    'summary': summaries_train
})

valid_dataset = Dataset.from_dict({
    'utterance': utterances_valid,
    'summary': summaries_valid
})

test_dataset = Dataset.from_dict({
    'utterance': utterances_test,
    'summary': summaries_test
})

# DatasetDict 생성
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': valid_dataset,
    'test': test_dataset
})

# DatasetDict 확인
print(dataset_dict)

# 토크나이저와 사전학습된 모델 불러오기
tokenizer_path = '/home/suyeon1803/PEGASUS_a100/PEGASUS_tokenizer2'
tokenizer = PegasusTokenizer(
    vocab_file=f"{tokenizer_path}/tokenizer.model",
    special_tokens_map_file=f"{tokenizer_path}/special_tokens_map.json",  
    tokenizer_config_file=f"{tokenizer_path}/tokenizer_config.json"  
)
model_checkpoint = "/home/suyeon1803/PEGASUS_a100/final_model" ######
model = PegasusForConditionalGeneration.from_pretrained(model_checkpoint)

# 데이터를 토큰화하는 함수
def preprocess_function(examples):
    model_inputs = tokenizer(examples['utterance'], truncation=True, padding='max_length', max_length=512)
    labels = tokenizer(examples['summary'], truncation=True, padding='max_length', max_length=128)
    
    model_inputs['labels'] = labels['input_ids']
    return model_inputs

# 데이터셋에 토큰화 적용
tokenized_dataset = dataset_dict.map(preprocess_function, batched=True)

# fine-tuning 학습 설정
training_args = TrainingArguments(
    output_dir="/home/suyeon1803/PEGASUS_a100/fine_tuned_model_check",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=4,
    num_train_epochs=5,
    weight_decay=0.01,
    save_steps=5000,
    save_total_limit=2,
    logging_dir="./logs",
)

# Trainer 객체 설정
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset['train'],
    eval_dataset=tokenized_dataset['validation'],
)

# 모델 학습 시작
trainer.train()

# 모델 저장 경로 지정
save_directory = "/home/suyeon1803/PEGASUS_a100/fine_tuned_model"

# 학습 완료 후 모델 저장
trainer.save_model(save_directory)
tokenizer.save_pretrained(save_directory)

# 테스트 데이터셋에서 평가 (test loss 확인)
test_results = trainer.evaluate(eval_dataset=tokenized_dataset['test'])
print(f"Test results: {test_results}")