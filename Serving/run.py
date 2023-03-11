from flask import Flask, render_template, request
from transformers import pipeline
from preprocessor import preprocess_sentence, preprocess_result, remove_empty_pattern
#from time_check import do_something
import math
import time
import sys
sys.path.append("C:\pythonStudy\추출요약\SRC")
from train import new_inference

test_from = "MODEL/KLUE/bert_transformer/model_step_1000.pt"
app = Flask(__name__)

@app.route('/', methods=['GET','POST'])

def home():
    #do_something()
    text_input = False
    text_input = str(request.form.get('size'))
    # 리스트 형식X 전체 데이터를 넣어서 '\n'.join으로 받기
    # 추출요약 모델 SRC 파일로 결과 출력까지 거의 그대로 구현
    text_input = "\n".join(text_input)
    text_output = new_inference(text_input, test_from, "transformer", "0", "0",1)
    return render_template('index.html', text_output=text_output)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(3000), debug=True)