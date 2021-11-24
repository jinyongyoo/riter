# RITER: Real-time Image Text Embedding Retreival
Riter connects image-text retreival methods from vision and NLP literature and Facebook's Faiss library to provide an easy way to build image-text retrieval systems. Riter also have pretrained models for joint image and text embeddings that users can use easily.

This started as a course project for CS 6501 Vision & Language course. It's currently on hold for now, but development will resume in the future. 

## Installation
First, clone this repo, `cd` into it, and run `pip install -e .` This should also install the required dependencies. 

Also, you'll need to install [Faiss](https://github.com/facebookresearch/faiss) package. Easiest way is to install via conda: `conda install faiss-cpu -c pytorch`. 

Lastly, for VSE++ models, you need to install resource for `nltk`, which is used for tokenization. Run the following in Python interpreter:
```
>>> import nltk
>>> nltk.download('punkt')
```

## Video Demo

[Demo](https://www.youtube.com/watch?v=Rp7vR6k1Vvg)
