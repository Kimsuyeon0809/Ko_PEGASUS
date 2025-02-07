from Korpora import KowikiTextKorpus

corpus = KowikiTextKorpus()
all_text = corpus.get_all_texts()


# 결과 출력 (처음 1000자만 출력해보기)
print(all_text[:2000])  # 너무 크기 때문에 일부만 출력


import re

def add_eod_to_documents(input_file_path, output_file_path):
    """
    주어진 텍스트 파일에서 = 문서 시작 구분자를 기준으로 문서를 나누고
    각 문서 끝에 <EOD> 토큰을 추가한 후 결과를 새로운 파일로 저장.
    
    Args:
    - input_file_path (str): 입력 텍스트 파일 경로
    - output_file_path (str): 출력 텍스트 파일 경로
    """
    # 텍스트 파일에서 데이터 읽기
    with open(input_file_path, 'r', encoding='utf-8') as file:
        data = file.read()

    # 문서의 시작을 하나의 =으로 감싸진 부분만을 기준으로 나누는 정규식 패턴
    documents = re.split(r'(^= [^=]+ =\s*)', data, flags=re.MULTILINE)

    # 각 문서 끝에 <EOD> 토큰을 추가하고 다시 합침
    documents_with_eod = []
    for i in range(1, len(documents), 2):
        document = documents[i] + documents[i+1].strip()  # 문서 제목과 내용을 결합
        documents_with_eod.append(document + "\n<EOD>")

    # 결과를 하나의 문자열로 합침
    final_text = "\n".join(documents_with_eod)

    # 결과를 새로운 파일로 저장
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(final_text)

    print(f"처리가 완료되었습니다. 결과는 {output_file_path}에 저장되었습니다.")

input_file_path = '/home/suyeon1803/PEGASUS_a100/kowikitext/kowikitext_20200920.dev'
output_file_path = '/home/suyeon1803/PEGASUS_a100/kowikitext/kowikitext_20200920.dev_'
add_eod_to_documents(input_file_path, output_file_path)
