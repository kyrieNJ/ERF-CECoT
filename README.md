# ERF-CECoT
Code for our paper:
An Empathetic Response Generation Framework Based on the Cognitive-Emotion Chain-of-Thought

## Requirements
* Python 3.10.18
* langchain 0.3.27
* torch 2.7.1
* SpaCy 3.7.5
* numpy 1.26.4
* scikit-learn 1.7.0
* sentencepiece 0.2.0
* transformers 4.45.1
* zh-core-web-sm 3.7.0
* jieba 0.42.1

## Train stage
* You can train the model for cognitive distortion detector, run the code [train4cd.py](./ERF-CECoT/pretrain/train4cd.py).
```bash
python ./pretrain/train4cd.py 
```
* You can train the model for emotion detector, run the code [train4emo.py](./ERF-CECoT/pretrain/train4emo.py).
```bash
python ./pretrain/train4emo.py 
```
* You can also use the trained model to infer the results for the given data, run the code [infer4cdemo.py](./ERF-CECoT/pretrain/infer4cdemo.py).
```bash
python ./pretrain/infer4cdemo.py 
```

## Citation
If our work has been helpful to you, please mark references to our work in your research and thank you for your support.

## Detailed information on few-shot samples
<img width="627" height="932" alt="image" src="https://github.com/user-attachments/assets/e44db537-4ed6-46ac-8f6f-3fcdae14697a" />
