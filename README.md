
## 💡프로젝트 소개

#### 1️⃣ 주제 : 텍스트 추출 요약<br>
#### 2️⃣ 설명 : [Fine-tune BERT for Extractive Summarization](https://arxiv.org/pdf/1903.10318v2.pdf)을 기반으로 추출 요약 모델 구현<br> 
#### 3️⃣ 모델 : Hugging Face [klue/bert-base](https://huggingface.co/klue/bert-base) 모델 사용하여 진행<br><br>

## 논문 소개
- BERT를 기반으로 Simple Classifier, Inter-sentence Transformer, Recurrent Neural Network 세가지 종류의 summarization-specific layers를 추가하여 추출 요약 실험 진행
<br>

![](img/bertsum.png)
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
## 1. train

```
logdirlocation = 'LOG/KLUE'
os.makedirs(logdirlocation, exist_ok=True)

!python SRC/train.py \
  -mode train \
  -encoder transformer \
  -dropout 0.1 \
  -bert_data_path data/bert_data/train/korean \
  -model_path MODEL/KLUE/bert_transformer \
  -lr 2e-3 \
  -visible_gpus 0 \
  -gpu_ranks 0 \
  -world_size 1 \
  -report_every 1000\
  -save_checkpoint_steps 100 \
  -batch_size 1000 \
  -decay_method noam \
  -train_steps 1000 \
  -accum_count 2 \
  -log_file LOG/KLUE/bert_transformer.txt \
  -use_interval true \
  -warmup_steps 200 \
  -ff_size 2048 \
  -inter_layers 2 \
  -heads 8
```

## 2. Test
```
!python SRC/train.py \
  -mode inference \
  -visible_gpus -1 \
  -gpu_ranks -1 \
  -world_size 0 \
  -log_file LOG/KLUE/bert_transformer.txt \
  -test_from MODEL/KLUE/bert_transformer/model_step_1000.pt \
  -input_text raw_data/valid/valid_0.txt
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
