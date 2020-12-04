import os
import pickle
import faiss
import PIL
import torch
from tqdm import tqdm
import gensim
import numpy as np
from collections import defaultdict

from riter import fields, utils


class SimilarityIndex:
    """
    Index that uses mutimodal embedding space of images and text.
    Similarity is measured using cosine similarity of the vectors.

    Args:
        ndim (int): The dimension size of the faiss index.
        schema (riter.Schema): Schema object.
    """

    def __init__(self, schema, index_recipe):
        self.schema = schema
        self.recipe = index_recipe

        for name in self.recipe:
            if name not in self.schema:
                raise ValueError(
                    f"Field {name} cannot be found in data schema. Please first add it to the data schema."
                )

        self._doc2idx = {}
        self._idx2doc = []

        self._indices = {}
        for fname in self.recipe.indexable_fields():
            if self.recipe[fname]["type"] == "faiss":
                self._indices[fname] = faiss.IndexFlatIP(self.recipe[fname]["ndim"])
            elif self.recipe[fname]["type"] == "gensim":
                self._indices[fname] = None
            else:
                raise ValueError(
                    f"Index type {self.recipe[fname]['type']} unavailable."
                )

    def build(self, documents, encoders, preprocessors, batch_size=32):
        utils.logger.info(f"Building index.")
        orig_n = len(documents)
        i = len(self._idx2doc)
        for d in documents:
            doc = utils.Document(d)
            if self.schema.check(doc):
                documents.append(doc)
                self._doc2idx[doc] = i
                i += 1

        self._idx2doc.extend(documents)

        if len(documents) != orig_n:
            utils.logger.info(
                f"Filtered out {orig_n-len(documents)} documents that does not match the schema."
            )

        doc_dataset = utils.DocDataset(documents, self.schema)
        for name in self.recipe.indexable_fields():
            utils.logger.info(f"Processing field '{name}'.")
            doc_dataset.set_field(name)
            vectors = []
            if self.recipe[name]["type"] == "faiss":
                if name not in encoders:
                    raise ValueError(f"Cannot find encoder for field '{name}'")
                if name not in preprocessors:
                    raise ValueError(f"Cannot find preprocessor for field '{name}'")
                dataloader = torch.utils.data.DataLoader(
                    doc_dataset,
                    batch_size=batch_size,
                    pin_memory=True,
                    num_workers=2,
                    collate_fn=preprocessors[name],
                )

                encoder = encoders[name]
                encoder.eval()
                encoder.to(utils.device)

                with torch.no_grad():
                    for inputs in tqdm(dataloader):
                        if isinstance(inputs, dict):
                            for k in inputs:
                                inputs[k] = inputs[k].to(utils.device)
                            outputs = encoder(**inputs)
                        elif isinstance(inputs, (list, tuple)):
                            inputs = list(inputs)
                            for i in range(len(inputs)):
                                inputs[i] = inputs[i].to(utils.device)
                            outputs = encoder(*inputs)
                        else:
                            inputs = inputs.to(utils.device)
                            outputs = encoder(inputs)

                        vectors.append(outputs.detach().cpu())

                vectors = torch.cat(vectors, dim=0).detach().numpy()
                faiss.normalize_L2(vectors)
                self._indices[name].add(vectors)

            elif self.recipe[name]["type"] == "gensim":
                if name not in preprocessors:
                    raise ValueError(f"Cannot find preprocessor for field '{name}'")
                analyzer = preprocessors[name]
                corpus = []
                for i in range(len(doc_dataset)):
                    text = doc_dataset[i]
                    words = analyzer(text)
                    corpus.append(words)
                dictionary = gensim.corpora.Dictionary(corpus)
                bow_corpus = [dictionary.doc2bow(c) for c in corpus]
                tfidf_model = gensim.models.TfidfModel(bow_corpus)
                self._indices[name] = (
                    dictionary,
                    tfidf_model,
                    gensim.similarities.SparseMatrixSimilarity(
                        bow_corpus, num_features=len(dictionary)
                    )
                )

        utils.logger.info(f"Finished building index.")

    def get(self, docid):
        return self._idx2doc[self._doc2idx[docid]]

    def search(
        self, query, encoders, preprocessors, top_k=100, score_weights={}, min_score=0.2
    ):
        """
        Search index with `query`
        Args:
            query (dict[str, object]): query is a dictionary where keys are field names and values are the query values.
            top_k (int): The top-K results to return for each field.
            score_weights (dict[str, float]): dictionary where keys are field names and values are weight values for each field.
        Returns:
            list[Documents]: list of relative documents.
        """
        if not score_weights:
            n = len(query.keys())
            for field_name in query:
                score_weights[field_name] = 1 / n
        total_scores = defaultdict(lambda: 0.0)
        for field_name in query:
            value = query[field_name]
            if self.recipe[field_name]["type"] == "faiss":
                if field_name not in encoders:
                    raise ValueError(f"Cannot find encoder for field '{field_name}'")
                if field_name not in preprocessors:
                    raise ValueError(
                        f"Cannot find preprocessor for field '{field_name}'"
                    )
                encoder = encoders[field_name]
                encoder.eval()
                preprocessor = preprocessors[field_name]
                value = preprocessor(value)
                with torch.no_grad():
                    if isinstance(value, dict):
                        query_vec = encoder(**value).detach().numpy()
                    elif isinstance(inputs, (list, tuple)):
                        query_vec = encoder(*value).detach().numpy()
                    else:
                        query_vec = encoder(value).detach().numpy()
                faiss.normalize_L2(query_vec)
                scores, indices = self._indices[field_name].search(query_vec, top_k)
                indices = indices[0]
                score_sum = np.sum(indices)
                scores = scores[0]
                results = zip(indices, scores)

            elif self.recipe[field_name]["type"] == "gensim":
                dictionary, model, index = self._indices[field_name]
                if field_name not in preprocessors:
                    raise ValueError(
                        f"Cannot find preprocessor for field '{field_name}'"
                    )
                words = preprocessors[field_name](value)
                query_bow = dictionary.doc2bow(words)
                results = list(enumerate(index[model[query_bow]]))
                score_sum = 0.0
                for idx, score in results:
                    score_sum += score

            for idx, score in results:
                total_scores[idx] += score_weights[field_name] * score / score_sum

            indices = sorted(indices, key=lambda k: total_scores[k], reverse=True)[
                :top_k
            ]
            scores = [total_scores[i] for i in indices]

        return [(self._idx2doc[i], score) for i, score in zip(indices, scores)]

    def save(self, path):
        """Save index to directory. This involves saving the Faiss indices as
        well as id mappings.

        Args:
            path (str): Path of directory to save the contents of this index.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        utils.logger.info(f"Saving index at {path}")

        for fname in self.recipe.indexable_fields():
            index_path = os.path.join(path, f"{fname}.index")
            if self.recipe[fname]["type"] == "faiss":
                faiss.write_index(self._indices[fname], index_path)
            elif self.recipe[fname]["type"] == "gensim":
                dictionary, model, index = self._indices[fname]
                dictionary.save(os.path.join(path, f"{fname}_gensim.dict"))
                model.save(os.path.join(path, f"{fname}_gensim.model"))
                index.save(os.path.join(path, f"{fname}_gensim.index"))

        doc2idx_path = os.path.join(path, "mapping.pkl")
        document_path = os.path.join(path, "documents.pkl")
        schema_path = os.path.join(path, "schema.pkl")
        recipe_path = os.path.join(path, "recipe.pkl")
        with open(document_path, "wb") as f:
            pickle.dump(self._idx2doc, f)
        with open(doc2idx_path, "wb") as f:
            pickle.dump(self._doc2idx, f)
        with open(schema_path, "wb") as f:
            pickle.dump(self.schema, f)
        with open(recipe_path, "wb") as f:
            pickle.dump(self.recipe, f)

    @classmethod
    def load(cls, path):
        """Load index from path

        Args:
            path (str): Path of directory where the contents of this index is stored.
        """
        index = cls.__new__(cls)
        if not os.path.exists(path):
            raise FileNotFoundError(f"'{path}' cannot be found.")

        utils.logger.info(f"Loading index from {path}")

        doc2idx_path = os.path.join(path, "mapping.pkl")
        document_path = os.path.join(path, "documents.pkl")
        schema_path = os.path.join(path, "schema.pkl")
        recipe_path = os.path.join(path, "recipe.pkl")
        with open(document_path, "rb") as f:
            index._idx2doc = pickle.load(f)
        with open(doc2idx_path, "rb") as f:
            index._doc2idx = pickle.load(f)
        with open(schema_path, "rb") as f:
            index.schema = pickle.load(f)
        with open(recipe_path, "rb") as f:
            index.recipe = pickle.load(f)

        index._indices = {}
        for fname in index.recipe.indexable_fields():
            index_path = os.path.join(path, f"{fname}.index")
            if index.recipe[fname]["type"] == "faiss":
                index._indices[fname] = faiss.read_index(index_path)
            elif index.recipe[fname]["type"] == "gensim":
                dictionary = gensim.corpora.Dictionary.load(
                    os.path.join(path, f"{fname}_gensim.dict")
                )
                model = gensim.models.TfidfModel.load(
                    os.path.join(path, f"{fname}_gensim.model")
                )
                matindex = gensim.similarities.SparseMatrixSimilarity.load(
                    os.path.join(path, f"{fname}_gensim.index")
                )
                index._indices[fname] = (dictionary, model, matindex)

        return index
