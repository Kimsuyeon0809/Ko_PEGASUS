import torch
from transformers import PegasusTokenizer, PegasusForConditionalGeneration
import re
import emoji
from transformers import PegasusForConditionalGeneration, AutoTokenizer

# Function to remove both text-based emoticons and emojis
def remove_emoticons_and_emojis(text):
    # Remove text-based emoticons using regex
    text = re.sub(r':\)|:\(|;\)|;\(|:D|:P|;D|:\||:\*|<3|:o|:O|:/|:-/|:\'|>:\(|@\)|:@|;@\)', '', text)
    # Remove emojis using the emoji library
    text = emoji.replace_emoji(text, replace='')
    
    return text

# Function to preprocess the text
def preprocess_text(text):
    # Remove emoticons and emojis
    text = remove_emoticons_and_emojis(text)
    
    # Remove unwanted characters, such as excessive punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Optionally, add more preprocessing techniques here
    
    return text


input_text = "난 낼도 약속 이써 누구만남 ㅋㅋㅋㅋ 아 같이 일하는 사람 너는 같이일하는 사람이랑 겁나 잘만난다 ㅋㅋ 그냥 뭐 일하다 보면 친해지니까 걍 밥 먹고 수다 떠는 거야 신기하네"# train set
# input_text = "진짜 웃기지 않냐. 응. 그래서 내가 마트에서 빵을 사왔는데 배가 아파서 못먹었어. 헐 내일 병원 가봐. 응 내일 가보려고."
# input_text= "레이나 모닝 사야곘어 맞아 경차 레이 이쁘네 #@URL# 싸다 와 이건 주행거리 거의 10만이야ㅋㅋㅋㅋㅋ ㅋㅋ글쿤 엄마가 사주면 좋겠따" # test set
# input_text = "ㅋㅋㅋ 렌트카 예약했어 모닝으로ㅋㅋㅋ 오 빨라 빨라!! 8시에 가두 된댕?ㅋㅋㅋㅋㅋ 얼마얌? 또 7만원 아니지?ㅋㅋ 모닝인데 7만원이면.. 양아치다 진짜 가격은 못물어봤엉ㅋㅋㅋ 다른거 물어보느라ㅜㅜㅜㅜㅜ 내일 출발할때 연락달라고하던데 아 ㅠ_ㅠ.... 그랫구나....ㅠㅠㅠㅠㅠㅠ 가격 똑같으면 다른걸로 빌리는게 낫것지? 아 내일 거기 가기 전에? 내일가기전에 연락하래 그래도경차가 까스덜먹으니깐 괜찮을껄 그렇군 ㅋㅋㅋㅋㅋ  하긴 들먹으니까 ㅋㅋㅋ"# train set
# input_text = '헐맞다 그러네? 그럼 밥먹기..? 나 점심 먹고나갈거야.. 그럼 그냥 롯데몰 돌아다니기..ㅋ 야 백화점도 아홉시까지밖에 안하는거알지? 아맞네ㅋㅋㅋㅋㅋㅋㅋ'
# input_text = '잠즐어서 좀 늦을 듯 운동 갔다가 집 들러서 케이크 갔다줄게 ㅠㅠ 아니면 안줘두댕 ㅋㅋ나오늘 9시 쯤 끝나서 괜찮아 ㅋㅋㅋㅋ 아냐ㅋㅋㅋㅋㅋ 갖다둘거야 근데 진짜 얼마 안 댐 먹어치워야댐 ㅋㅋㅋㅋ 나 9시 전에는 가 고뤠 얼른 와'
# input_text = '예약잡고 싶다고 얘기하면 알아서 안내해줄듯 ㄲ아으악 떨린당!! 이번달 말로 예약 가능한지 물어봤엉ㅎㅎㅎ 끼야옹 끼아아능ㄱ 날짜랑 시간 잡은즁'
input_text = preprocess_text(input_text) # 입력 텍스트 전처리 

# Print the preprocessed text
print("Preprocessed Text:", input_text)
# 체크포인트 모델 경로 설정
checkpoint_path = "/home/suyeon1803/PEGASUS_a100/new_pretrain_checkpoing/checkpoint-30000"
tokenizer_path = '/home/suyeon1803/PEGASUS_a100/PEGASUS_tokenizer'

# 모델 및 토크나이저 로드 (GPU로 이동)
tokenizer = PegasusTokenizer(
    vocab_file=f"{tokenizer_path}/tokenizer.model",
    special_tokens_map_file=f"{tokenizer_path}/special_tokens_map.json",  
    tokenizer_config_file=f"{tokenizer_path}/tokenizer_config.json"  
)
model = PegasusForConditionalGeneration.from_pretrained(checkpoint_path)

# 입력 텍스트를 토큰화
inputs = tokenizer(input_text, return_tensors="pt", padding="longest", truncation=True, max_length=256)

# 모델을 통해 출력 생성 (요약 또는 텍스트 생성)
with torch.no_grad():  # 평가 시에는 그래디언트 계산을 하지 않음
    summary_ids = model.generate(inputs['input_ids'], max_length=50, num_beams=5, early_stopping=True)

# 출력된 요약 결과를 디코딩하여 읽을 수 있는 텍스트로 변환
output_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# 결과 출력
print("입력 텍스트:", input_text)
print("생성된 출력 텍스트:", output_text)
