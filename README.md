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
* You can train the model, run the code [train4cd.py](./ERF-CECoT/pretrain/train4cd.py).
```bash
python ./SM-HK/train.py 
```
* You can also use the trained model to infer the results for the given data, run the code [infer.py](./SM-HK/infer.py).
```bash
python ./SM-HK/infer.py 
```

## Citation

If our work has been helpful to you, please mark references to our work in your research and thank you for your support.
