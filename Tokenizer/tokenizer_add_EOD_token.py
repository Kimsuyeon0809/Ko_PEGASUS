import os
import torch
from transformers import PegasusTokenizer

# 학습된 SentencePiece 토크나이저 경로 (파일들이 있는 경로를 지정하세요)
tokenizer_path = '/home/suyeon1803/PEGASUS_a100/PEGASUS_tokenizer2'

# 로컬에 저장된 SentencePiece 토크나이저 불러오기
tokenizer = PegasusTokenizer(
    vocab_file=f"{tokenizer_path}/tokenizer.model",
    special_tokens_map_file=f"{tokenizer_path}/special_tokens_map.json",  
    tokenizer_config_file=f"{tokenizer_path}/tokenizer_config.json"  
)

print("로컬 SentencePiece 토크나이저 로드 완료")

# 먼저, 추가할 스페셜 토큰을 정의합니다.

special_tokens_dict = {'additional_special_tokens': ["<EOD>"]}

# 토크나이저에 스페셜 토큰 추가
num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
tokenizer.save_pretrained(tokenizer_path)

# 새로운 토큰이 추가되었는지 확인
print(f"Added {num_added_tokens} new tokens: {tokenizer.additional_special_tokens}")
