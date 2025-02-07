import torch
import nltk
import random

# 2. EOD와 EOD 사이 txt의 문장 중 30%를 [MASK1]로 대체
def apply_gsg_to_documents(eod_data, mask_ratio=0.3):
    # EOD 토큰을 기준으로 문서 분리
    documents = eod_data.split('<EOD>')

    masked_documents = []
    original_masked_sentences = []

    # 각 문서에 대해 GSG 적용
    for document in documents:
        document = document.strip()  # 문서 앞뒤 공백 제거
        
        if not document:
            continue
        
        sentences = nltk.sent_tokenize(document)  # 문서에서 문장 분리
        num_to_mask = max(1, int(len(sentences) * mask_ratio))  # 마스킹할 문장 수
        masked_sentences = random.sample(sentences, num_to_mask)  # 마스킹할 문장 선택

        # 마스킹할 문장을 <MASK1>로 대체
        masked_document = document
        for sentence in masked_sentences:
            masked_document = masked_document.replace(sentence, '[MASK1]')
        
        # 마스킹된 문서와 마스킹된 문장을 각각 저장
        masked_documents.append(masked_document.strip())  # 마스킹된 문서
        original_masked_sentences.append(' '.join(masked_sentences))  # 마스킹된 문장들
    
    # 최종적으로 모든 문서를 <EOD>로 다시 결합
    final_masked_data = ' <EOD> '.join(masked_documents) + ' <EOD>'
    
    return final_masked_data, original_masked_sentences

# 마스킹 할 텍스트 경로
with open('/home/suyeon1803/PEGASUS_a100/data/kiwiki_train_EOD.txt', 'r', encoding='utf-8') as f:
    eod_data = f.read()  # 파일 전체를 하나의 문자열로 읽어오기

masked_data, masked_sentences = apply_gsg_to_documents(eod_data)

# 마스킹된 텍스트 데이터 저장할 경로
with open('/home/suyeon1803/PEGASUS_a100/data/masked_kiwiki_train_EOD.txt', 'w', encoding='utf-8') as f:
    f.write(masked_data)
print('마스킹 완료')

# 위와 같은 경로 입력해서 잘 마스킹 되었는지 확인
with open('/home/suyeon1803/PEGASUS_a100/data/masked_kiwiki_train_EOD.txt', 'r', encoding='utf-8') as f:
    eod_data = f.read() 
print(eod_data[:100000])
print(len(eod_data))