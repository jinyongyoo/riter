import os
import pickle
import argparse
from functools import partial
from collections import defaultdict
import json

import torch
import torchvision
from pycocotools.coco import COCO
from PIL import Image
import nltk
from tqdm import tqdm

from riter import (
    AutoModel,
    AutoTokenizer,
    AutoTransformation,
    Schema,
    IndexRecipe,
    SimilarityIndex,
)
from riter.fields import ID, Text, ImagePath


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--url-path",
        default="./data/sbu/SBU_captioned_photo_dataset_urls.txt",
        type=str
    )
    parser.add_argument(
        "--caption-path",
        default="./data/sbu/SBU_captioned_photo_dataset_captions.txt",
        type=str
    )
    parser.add_argument(
        "--sample-size",
        default=10000,
        type=int
    )
    parser.add_argument(
        "--batch-size", default=128, type=int, help="Batch size of processing data"
    )
    parser.add_argument(
        "--index-save-path",
        default="./saved/sbu",
        type=str,
        help="Path to save index",
    )
    args = parser.parse_args()

    # First, come up with a schema for COCO images.
    schema = Schema(
        {
            "id": ID,
            "image_url": ID,
            "image": ImagePath(indexed=True),
            "caption": Text(indexed=True),
        }
    )

    with open(args.url_path, "r") as f:
        urls = f.readlines()
    with open(args.caption_path, "r") as f:
        captions = f.readlines()

    img_dir = "./data/sbu/images"
    N = args.sample_size
    i = 0
    print("Downloading images...")
    pbar = tqdm(total=N)
    documents = []
    # while len(documents) < N:
    #     url = urls[i]
    #     caption = captions[i]
    #     filepath = d + "/" + url.split("/")[-1].replace("\n", "")
    #     try:
    #         Image.open(io.BytesIO(requests.get(url).content)).save(filepath)
    #         document = {"id": url, "image_url": url, "image": filepath, "caption": caption}
    #         documents.append(document)
    #         pbar.update(1)
    #     except:
    #         pass
    #     i+=1
    # pbar.close()    

    with open("./data/sbu/sbu_dataset.json", "r") as f:
        documents = json.load(f)
    for i in range(len(documents)):
        documents[i]["id"] = documents[i]["image_url"]
        documents[i]["image"] = documents[i]["image_path"]
        del documents[i]["image_path"]
    
    # Now, decide how to index.
    recipe = IndexRecipe()
    # Images should be indexed using Faiss L2 distance index with dimension of 1024
    recipe.add_faiss_index_recipe("image", 1024)
    recipe.add_gensim_index_recipe("caption")

    # Create our index
    index = SimilarityIndex(schema, recipe)

    # Load our models from riter.zoo
    img_encoder = AutoModel.from_pretrained("vsepp-resnet-flickr")
    transformation = AutoTransformation.from_pretrained("vsepp-resnet-flickr")

    stop_words = set(nltk.corpus.stopwords.words("english"))
    stemmer = nltk.stem.PorterStemmer()

    def analyzer(text):
        # tokenize
        words = nltk.word_tokenize(text)
        # stem words
        words = [stemmer.stem(w) for w in words]
        # remove stop words
        words = [w for w in words if not w in stop_words]
        return words

    index.build(
        documents,
        {"image": img_encoder},
        {"image": transformation, "caption": analyzer},
        batch_size=args.batch_size,
    )

    index.save(args.index_save_path)
