from flask import Flask
from flask import render_template, request, send_from_directory
import json
import os
import pickle
import nltk
from pycocotools.coco import COCO

from riter import SimilarityIndex, AutoModel, AutoTokenizer, AutoTransformation


INDEX_PATH_DIR = "saved/mscoco"
DEBUG = True


text_encoder = AutoModel.from_pretrained("vsepp-gru-coco")
tokenizer = AutoTokenizer.from_pretrained("vsepp-gru-coco")
val_index = SimilarityIndex.load(os.path.join(INDEX_PATH_DIR, "val_gensim_index"))
train_index = SimilarityIndex.load(os.path.join(INDEX_PATH_DIR, "train_gensim_index"))

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
    @app.route("/coco", methods=["GET"])
    def coco_joint_query():
        data = request.args
        query = data.get("query", "dog playing with ball")
        data_split = data.get("split")
        page_num = data.get("page_num", 1, type=int)
        page_len = data.get("page_len", 20, type=int)

        if data_split is None:
            data_split = "validation"

        if data_split == "validation":
            index = val_index
        else:
            index = train_index

        results = index.search(
            {"image": query, "caption": query},
            {"image": text_encoder},
            {"image": tokenizer, "caption": analyzer},
            score_weights={"image": 0.7, "caption": 0.3},
            top_k=20,
            min_score=0.5,
        )

        results = [r[0] for r in results]

        return render_template(
            "coco.html", query=query, results=results, split=data_split
        )

    return app


# run the app.
if __name__ == "__main__":
    app = create_app()
    app.run(debug=DEBUG, host="0.0.0.0", port=9999)
