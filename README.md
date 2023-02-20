
## 💡프로젝트 소개
```
1️⃣ 주제 : 텍스트 추출 요약
2️⃣ 설명 : [Fine-tune BERT for Extractive Summarization](https://arxiv.org/pdf/1903.10318v2.pdf)을 기반으로 추출 요약 모델 구현 
3️⃣ 모델 : Hugging Face [klue/bert-base](https://huggingface.co/klue/bert-base) 모델 사용하여 진행
```
## 논문 소개
- pre-training 모델을 post-training를 통해서 도메인 적응을 하고 fine-tuning를 진행해 성능 향상을 기대한다.
- fine-grained : 소량의 후보에서 최적의 후보를 선택하는 방법으로 주로 One-tower 구조의 모델을 구현하여 성능을 향상시킨다.
![](img/mrs.png)
<Br><br>
### 부연설명
- post-training
  - 전체 대화를 여러 개의 short context-response pairs로 나누어 모델을 학습
    - candidate을 positive, random negative, context negative로 3개 class로 구성해서 학습
    - 이를 통해 발화 관련 분류(URC)로 발화간의 관계 및 발화 내적 관계를 배워 데이터 증강과 성능 향상의 효과를 얻는다. 
  - MLM 사용

---
## 1. post-training & fine-tuning

```
!pip install transformers==4.25.1


!python3 post_pretrain/train.py
!python3 fine_tuning/train.py
```

## 2. Test
```
import torch
from model import FineModel
fine_model = FineModel().cuda()
fine_model.load_state_dict(torch.load('/content/drive/MyDrive/인공지능/멀티턴응답선택/fine_model.bin'))


context = ["어떤 문제가 있으신가요?", "어떤 차를 사야 할지 잘 모르겠어요.", "차는 한 번 사면 10년도 넘게 써서, 신중하게 골라야 해요."]
candidates = ["저 좀 도와주실 수 있나요?", "저는 농구를 좋아합니다.", "자동차는 신중히 골라야합니다.", "저는 차 마시는거를 좋아합니다.", "날씨가 화창합니다.", "저는 차를 좋아합니다."]
     

context_token = [fine_model.tokenizer.cls_token_id]
for utt in context:
    context_token += fine_model.tokenizer.encode(utt, add_special_tokens=False)
    context_token += [fine_model.tokenizer.sep_token_id]

session_tokens = []    
for response in candidates:
    response_token = [fine_model.tokenizer.eos_token_id]
    response_token += fine_model.tokenizer.encode(response, add_special_tokens=False)
    candidate_tokens = context_token + response_token        
    session_tokens.append(candidate_tokens)
    
# 최대 길이 찾기 for padding
max_input_len = 0
input_tokens_len = [len(x) for x in session_tokens]
max_input_len = max(max_input_len, max(input_tokens_len))    
    
batch_input_tokens = []
batch_input_attentions = []
for session_token in session_tokens:
    input_token = session_token + [fine_model.tokenizer.pad_token_id for _ in range(max_input_len-len(session_token))]
    input_attention = [1 for _ in range(len(session_token))] + [0 for _ in range(max_input_len-len(session_token))]
    batch_input_tokens.append(input_token)
    batch_input_attentions.append(input_attention)
    
batch_input_tokens = torch.tensor(batch_input_tokens).cuda()
batch_input_attentions = torch.tensor(batch_input_attentions).cuda()


softmax = torch.nn.Softmax(dim=1)
results = fine_model(batch_input_tokens, batch_input_attentions)
prob = softmax(results)
true_prob = prob[:,1].tolist()

print(context)
for utt, prob in zip(candidates, true_prob):
    print(utt, '##', round(prob,3))
```

---
## 🗓️ 프로젝트 개선 진행

|개선 서비스|진행사항(%)|
|:----------:|:------:|
|데이터셋 추가 확보||
|키워드에 따라 bias되는 문제 연구||
|다화자의 경우 고려||
|발화의 추가 특징 고려||
|context 길이 선정||
|같은 세션을 판단할 모듈있는지 고려||
|teacher 모델의 예측 값을 활용하여 학습|


---
