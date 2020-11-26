import os
import pickle
import argparse
from functools import partial
from collections import defaultdict

import torch
import torchvision
from pycocotools.coco import COCO
from PIL import Image
import nltk

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
        "--img-data-path",
        default="./data/mscoco/imgs/val2017",
        type=str,
        help="Path to image mscoco dataset",
    )
    parser.add_argument(
        "--annotation-data-path",
        default="./data/mscoco/annotations/captions_val2017.json",
        type=str,
        help="Path to JSON file for annotations",
    )
    parser.add_argument(
        "--batch-size", default=128, type=int, help="Batch size of processing data"
    )
    parser.add_argument(
        "--index-save-path",
        default="./saved/mscoco/val_index",
        type=str,
        help="Path to save index",
    )
    args = parser.parse_args()

    # First, come up with a schema for COCO images.
    schema = Schema(
        {
            "id": ID,
            "url": Text(indexed=False),
            "image": ImagePath(indexed=True),
            "caption": Text(indexed=True),
        }
    )

    coco = COCO(args.annotation_data_path)
    # List of images we want to index
    img_ids = list(coco.imgs.keys())
    documents = []
    for i in img_ids:
        url = coco.imgs[i]["coco_url"]
        anno_ids = coco.getAnnIds(imgIds=[i])
        caption = " ".join([coco.anns[j]["caption"] for j in anno_ids])
        img_path = os.path.join(args.img_data_path, coco.loadImgs(i)[0]["file_name"])
        # Documents can be represented by a simple dictionary where keys match with the keys in the schema
        documents.append({"id": i, "url": url, "image": img_path, "caption": caption})

    # Now, decide how to index.
    recipe = IndexRecipe()
    # Images should be indexed using Faiss L2 distance index with dimension of 1024
    recipe.add_faiss_index_recipe("image", 1024)
    recipe.add_gensim_index_recipe("caption")

    # Create our index
    index = SimilarityIndex(schema, recipe)

    # Load our models from riter.zoo
    img_encoder = AutoModel.from_pretrained("vsepp-resnet-coco")
    transformation = AutoTransformation.from_pretrained("vsepp-resnet-coco")

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
