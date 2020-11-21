from flask import Flask
from flask import render_template, request, send_from_directory
import json
import os
import pickle
import torch
from pycocotools.coco import COCO

from riter import JointEmbeddingIndex, AutoModel, AutoTokenizer, AutoTransformation


INDEX_PATH_DIR = "saved/mscoco"
VAL_JSON_PATH = "data/mscoco/annotations/captions_val2017.json"
TRAIN_JSON_PATH = "data/mscoco/annotations/captions_train2017.json"

DEBUG = True

class CocoDataset:
    """COCO Custom Dataset

    Args:
        img_root: path of image directory
        annotation_json: path of JSON file for COCO annotations
    """

    def __init__(self, annotation_json):
        self.coco = COCO(annotation_json)

    def __getitem__(self, img_id):
        anno_ids = self.coco.getAnnIds(img_ids=[img_id])
        img_file = os.path.join(
            self.img_root, self.coco.loadImgs(img_id)[0]["file_name"]
        )

        text = " ".join(
            [self.coco.anns[i]["caption"] for i in anno_ids]
        )

        return img_id, text


def create_index(index_path):
    img_encoder = AutoModel.from_pretrained("vsepp-resnet-coco")
    transformation = AutoTransformation.from_pretrained("vsepp-resnet-coco")
    text_encoder = AutoModel.from_pretrained("vsepp-gru-coco")
    tokenizer = AutoTokenizer.from_pretrained("vsepp-gru-coco")
    index = JointEmbeddingIndex(
        1024, img_encoder, transformation, text_encoder, tokenizer
    )
    index.load(index_path)
    return index


def create_app():
    app = Flask(
        __name__, template_folder="templates", static_folder="static"
    )

    val_index = create_index(os.path.join(INDEX_PATH_DIR, "val_index"))
    train_index = create_index(os.path.join(INDEX_PATH_DIR, "train_index"))
    val_coco = COCO(VAL_JSON_PATH)
    train_coco = COCO(TRAIN_JSON_PATH)

    # Actually run search
    @app.route("/")
    @app.route("/coco", methods=["GET"])
    def coco_joint_query():
        data = request.args
        query = data.get("query", "dog playing with ball")
        data_split = data.get("split")
        page_num = data.get("page_num", 1, type = int)
        page_len = data.get("page_len", 20, type = int)

        if data_split is None:
            data_split = "validation"

        top_k = 20
        if data_split == "validation":
            results = val_index.query(query, top_k=top_k)
            coco = val_coco
        else:
            results = train_index.query(query, top_k=top_k)
            coco = train_coco
        
        query_results = []
        print(results)
        for img_id in results:
            anno_ids = coco.getAnnIds(imgIds=[img_id])
            img_url = coco.imgs[img_id]["coco_url"]
            captions = [coco.anns[i]["caption"] for i in anno_ids]
            query_results.append({"img_url": img_url, "captions": captions})

        return render_template(
            "coco.html", 
            query=query, 
            results=query_results,
            split=data_split    
        )

    return app


# run the app.
if __name__ == "__main__":
    app = create_app()
    app.run(debug=DEBUG, host="0.0.0.0", port=9999)
