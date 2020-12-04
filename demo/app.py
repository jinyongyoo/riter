from flask import Flask
from flask import render_template, request, send_from_directory
import json
import os
import pickle
import nltk
from pycocotools.coco import COCO

from riter import SimilarityIndex, AutoModel, AutoTokenizer, AutoTransformation


COCO_INDEX_PATH_DIR = "saved/mscoco"
SBU_INDEX_PATH_DIR = "saved/sbu"
DEBUG = True


coco_text_encoder = AutoModel.from_pretrained("vsepp-gru-resnet-coco")
coco_tokenizer = AutoTokenizer.from_pretrained("vsepp-gru-resnet-coco")
coco_val_index = SimilarityIndex.load(os.path.join(COCO_INDEX_PATH_DIR, "val_gensim_index"))
coco_train_index = SimilarityIndex.load(os.path.join(COCO_INDEX_PATH_DIR, "train_gensim_index"))
sbu_text_encoder = AutoModel.from_pretrained("vsepp-gru-resnet-flickr")
sbu_tokenizer = AutoTokenizer.from_pretrained("vsepp-gru-resnet-flickr")
sbu_index = SimilarityIndex.load(SBU_INDEX_PATH_DIR)

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

def create_app():
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Actually run search
    @app.route("/")
    @app.route("/index")
    @app.route("/coco", methods=["GET"])
    def coco_joint_query():
        data = request.args
        query = data.get("query", "dog playing with ball")
        data_split = data.get("split")

        if data_split is None:
            data_split = "validation"

        if data_split == "validation":
            index = coco_val_index
        else:
            index = coco_train_index

        results = index.search(
            {"image": query, "caption": query},
            {"image": coco_text_encoder},
            {"image": coco_tokenizer, "caption": analyzer},
            score_weights={"image": 0.7, "caption": 0.3},
            top_k=20,
            min_score=0.5,
        )

        results = [r[0] for r in results]

        return render_template(
            "coco.html", query=query, results=results, split=data_split
        )

    # Actually run search
    @app.route("/sbu", methods=["GET"])
    def sbu_joint_query():
        data = request.args
        query = data.get("query", "dog playing with ball")


        results = sbu_index.search(
            {"image": query, "caption": query},
            {"image": sbu_text_encoder},
            {"image": sbu_tokenizer, "caption": analyzer},
            score_weights={"image": 0.7, "caption": 0.3},
            top_k=20,
            min_score=0.5,
        )

        results = [r[0] for r in results]

        return render_template(
            "sbu.html", query=query, results=results,
        )
        return render_template("sbu.html")

    return app


# run the app.
if __name__ == "__main__":
    app = create_app()
    app.run(debug=DEBUG, host="0.0.0.0", port=9999)
