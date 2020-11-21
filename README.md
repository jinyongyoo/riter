# RITER: Real-time Image Text Embedding Retreival
Riter connects image-text retreival methods from vision and NLP literature and Facebook's Faiss library to provide an easy way to build image-text retrieval systems. Riter also have pretrained models that makes it easy for users use. 

## Installation
Please run `pip install -r requirements.txt` to install required dependencies.

Also, you'll need to install [Faiss](https://github.com/facebookresearch/faiss) package. Easiest way is to install via conda: `conda install faiss-cpu -c pytorch`. 

Lastly, for VSE++ models, you need to install resource for `nltk`, which is used for tokenization. Run the following in Python interpreter:
```
>>> import nltk
>>> nltk.download('punkt')
```

## Getting Started


