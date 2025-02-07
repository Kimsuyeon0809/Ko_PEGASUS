
import torch
# from Korpora import Korpora
from transformers import PegasusConfig, PegasusTokenizer, PegasusForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
import json
from transformers import BertTokenizer
from transformers import AutoTokenizer, PegasusTokenizerFast
import os  

merged_data_path = f"/home/suyeon1803/PEGASUS_a100/data/kowiki/masked_preprocessed_kowiki_part1.txt"

# 파일 불러오기
with open(merged_data_path, 'r', encoding='utf-8') as file:
    train_dataset = file.read()

# 파일 내용 출력
print(train_dataset[:500])  # 파일의 처음 500글자 출력


# 학습된 SentencePiece 토크나이저 경로 
tokenizer_path = '/home/suyeon1803/PEGASUS_a100/PEGASUS_tokenizer'

# 로컬에 저장된 SentencePiece 토크나이저 불러오기
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

tokenizer = PegasusTokenizer(
    vocab_file=f"{tokenizer_path}/tokenizer.model",
    special_tokens_map_file=f"{tokenizer_path}/special_tokens_map.json",  
    tokenizer_config_file=f"{tokenizer_path}/tokenizer_config.json"  
)

print("로컬 SentencePiece 토크나이저 로드 완료")

def tokenize_data(data): # 수정 1: 토큰화 과정에서 진행상황 로깅할 수 있게 한 코드 
    # 데이터가 큰 경우 반복하면서 토큰화 진행
    total_data_size = len(data)
    tokenized_data = []

    for idx, sample in enumerate(data):
        # 입력 텍스트를 토큰화
        inputs = tokenizer(
            sample,
            truncation=True,
            padding="max_length",
            max_length=128
        )
        # labels를 입력과 동일하게 설정
        inputs['labels'] = inputs['input_ids'].copy()

        tokenized_data.append(inputs)

        # 진행 상황 출력
        if (idx + 1) % 1000000 == 0 or (idx + 1) == total_data_size:  # 1000000개마다 출력 또는 마지막에 출력
            percent_complete = (idx + 1) / total_data_size * 100
            print(f"{percent_complete:.2f}% 토큰화 완료")
            

    return tokenized_data

train_dataset = train_dataset.splitlines()  # txt 파일 데이터를 리스트로 분할 ##수정2 : 로깅 위해 분할

# 토큰화 진행
tokenized_data = tokenize_data(train_dataset) ## 수정3 : 맵핑 없이 txt 데이터 바로 사용 

print("토큰화 결과 저장 시작")
torch.save(tokenized_data, f'/home/suyeon1803/PEGASUS_a100/data/final_pt/masked_preprocessed_kowiki_part1.pt') # 불러오기 : tokenized_data = torch.load('tokenized_data.pt')
print("토큰화 결과 저장 완료")

print(tokenized_data[0])
print(len(tokenized_data))


