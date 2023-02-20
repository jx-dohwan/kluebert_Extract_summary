
## 💡프로젝트 소개
```
1️⃣ 주제 : 텍스트 추출 요약
2️⃣ 설명 : [Fine-tune BERT for Extractive Summarization](https://arxiv.org/pdf/1903.10318v2.pdf)을 기반으로 추출 요약 모델 구현 
3️⃣ 모델 : Hugging Face [klue/bert-base](https://huggingface.co/klue/bert-base) 모델 사용하여 진행
```
## 논문 소개
- BERT를 기반으로 Simple Classifier, Inter-sentence Transformer, Recurrent Neural Network 세가지 종류의 summarization-specific layers를 추가하여 추출 요약 실험 진행
<br>

![](img/bertsum.png)
<Br>
### 부연설명
- Embedding Multiple Sentences
  - 문장의 시작 : [CLS], 문장의 끝 : [SEP] 을 삽입하여 기존 [SEP]만 사용하여 문장들을 구분하던 BERT모델을 개선했다.
  - 여러개의 [CLS] 토큰을 사용하여 각 문장들의 feature를 [CLS] 토큰에 저장한다.
- Interval segment Embedding
  - 여러 문장이 포함된 문장에서 문서를 구분하기 위해서 사용됨
  - 요약 문서의 특성 상 두개 이상의 문장이 포함되므로 기본의 방식과는 다르게 문장1~4를 A와 B를 번갈아가며 구분
- Summarization Layers
  - BERT로부터 문장 vector에 대한 정보를 얻은 다음에 추출 요약을 위하여 문서 단위의 feature를 잡기 위해 그 결괏값 위에 summarization-specific layers를 쌓는다. 
  - Simple Classifier
    - 기존 BERT와 같이 Linear layer 및 Sigmoid function
  - Inter-sentence Transformer
    - 문장 representations을 위하여 Transformer layer을 사용하며 마지막 단계에서는 Transformer layer로부터 나온 문장 vector를 sigmoid classifier 넣는다. 그리고 Layer가 2개일 때의 성능이 제일 좋았다.
  - Recurrent Neural Network
    - Transformer와 RNN결합시 성능이 좋았으며 BERT output을 LSTM layer로 넘겨준다. 마지막 단계에서는 sigmoid classifier를 사용한다.


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

|개선사항|이유|진행률(%)|
|:-----:|:-----:|:-----:|
|Data Augmentation|법률문서 낮은 score||
|5만 step로 학습|테스트로 1000만 학습||
|Transformer로 서비스 구현|Transformer가 가장 성능이 좋음||
|RoBERTa, ELECTRA등 고려|BERT보다 좋은 성능 모델 존재||


---
