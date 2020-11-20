from flask import Flask
from flask import render_template, request, send_from_directory
import json
import os
import pickle
import torch

from riter import JointEmbeddingIndex
from riter.prebuilt import Tokenizer, ImageEncoder, TextEncoder
from riter.prebuilt.config import COCO_GRU_CONFIG, COCO_RESNET_CONFIG

from dataset import CocoDataset

ARTIFACT_ROOT = "/p/qdata/jy2ma/riter/RITER/saved/mscoco"
IMG_DATA_ROOT = "static/images/mscoco/val2017"
ANNO_JSON_PATH = (
    "/p/qdata/jy2ma/riter/RITER/data/mscoco/annotations/captions_val2017.json"
)
DEBUG = True


def create_app():
    app = Flask(
        __name__, template_folder="templates", static_folder="static"
    )

    img_encoder = ImageEncoder(**COCO_RESNET_CONFIG)
    img_encoder.load_state_dict(
        torch.load(os.path.join(ARTIFACT_ROOT, "coco_resnet.pth"))
    )
    text_encoder = TextEncoder(**COCO_GRU_CONFIG)
    text_encoder.load_state_dict(
        torch.load(os.path.join(ARTIFACT_ROOT, "coco_gru.pth"))
    )
    tokenizer = Tokenizer.load(os.path.join(ARTIFACT_ROOT, "coco_tokenizer.pkl"))
    search_index = JointEmbeddingIndex(
        1024, img_encoder, ImageEncoder.TRANSFORM, text_encoder, tokenizer
    )
    search_index.load(os.path.join(ARTIFACT_ROOT, "val_index"))
    coco_dataset = CocoDataset(IMG_DATA_ROOT, ANNO_JSON_PATH)

    @app.route("/results/<path:filepath>')/")
    def serve_img(filepath):
        return send_file(filepath)

    @app.route("/")
    @app.route("/index")
    def index():
        return render_template("index.html")

    # Actually run search
    @app.route("/query", methods=["GET"])
    def query():
        data = request.args
        query = data["query"]

        results = search_index.query(query, top_k=20)
        results = [coco_dataset[_id] for _id in results]
        
        return render_template("result.html", orig_query=query, results=results)

    return app


# run the app.
if __name__ == "__main__":
    app = create_app()
    app.run(debug=DEBUG, host="0.0.0.0", port=9999)
