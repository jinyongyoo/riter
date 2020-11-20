from abc import ABC, abstractmethod
import os
import pickle
import faiss
import PIL
import torch
from tqdm import tqdm
import numpy as np

from .utils import logger
import gc


class BaseIndex:
    """Base class representing an retreival index."""

    @abstractmethod
    def build(self):
        raise NotImplementedError()

    @abstractmethod
    def query(self):
        raise NotImplementedError()

    @abstractmethod
    def save(self, path):
        raise NotImplementedError()

    @abstractmethod
    def load(self, path):
        raise NotImplementedError()


class JointEmbeddingIndex(BaseIndex):
    """
    Index for the joint embedding space of images and text.
    It is composed of two faiss indices: (1) image index, (2) text index.
    Image index contains the vector representations of images, while text index contains the vector representation of text/captions.
    The idea is that given a query (e.g. string), we query both indices and combine the results.
    Similarity is measured using cosine similarity of the vectors.

    Args:
        dim (int): The dimension size of the faiss index.
        img_encoder (torch.nn.Module): The model for encoding image into a dense vector representation.
        transformation (torchvision.Transforms): The transformation steps applied to PIL.Image.Image objects to turn them into tensors.
        text_encoder (torch.nn.Module): The model for encoding text into a dense vector representation.
        tokenizer (obj): Tokenizer that when called (via `__call__`), returns list of ids that could be converted into a tensor and passed
            to the `text_encoder`.
    """

    def __init__(self, dim, img_encoder, transformation, text_encoder, tokenizer):
        self.dim = dim
        self.img_encoder = img_encoder
        self.transformation = transformation
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        self._guid2idx = {}
        self._idx2guid = []
        self._img_index = faiss.IndexFlatIP(self.dim)
        self._text_index = faiss.IndexFlatIP(self.dim)

        self.img_encoder.eval()
        self.text_encoder.eval()

    def _process_data(self, dataset, batch_size, collate_fn, device):
        if not isinstance(dataset, torch.utils.data.Dataset):
            raise ValueError("`dataset` must be of type `torch.utils.data.Dataset`.")

        if len(dataset) <= 0:
            raise ValueError("`dataset` must have size greater than 0.")

        self.img_encoder.to(device)
        self.text_encoder.to(device)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn
        )
        i = 0
        img_vecs = []
        text_vecs = []

        logger.info(f"Processing dataset of size {len(dataset)}.")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                # Run basic checks
                if len(batch) < 2:
                    raise ValueError(
                        "Tuple returned by `dataset` must contain at least two elements."
                    )
                if not (isinstance(batch[0][0], int) or isinstance(batch[0][0], str)):
                    raise ValueError(
                        "First element of tuple returned by `dataset` must be an integer or a string (ID must be hashable)"
                    )

                ids = batch[0]
                for _id in ids:
                    self._guid2idx[_id] = i
                    self._idx2guid.append(_id)
                    i += 1

                text_input = None
                if isinstance(batch[1], torch.Tensor) and len(batch[1][0].shape) >= 2:
                    # `batch[1]` is an image tensor
                    img = batch[1].to(device)
                    img_vec = self.img_encoder(img)
                    img_vecs.append(img_vec.detach().cpu())
                else:
                    # `batch[1]` is an token tensor
                    text_input = batch[1]

                if len(batch) == 3:
                    # If `batch` has three elements, then third element is token tensor
                    text_input = batch[2]

                if text_input is not None:
                    # Handle different formats of `text_input`
                    if isinstance(text_input, dict):
                        for k in text_input:
                            if isinstance(text_input[k], torch.Tensor):
                                text_input[k] = text_input[k].to(device)

                        text_vec = self.text_encoder(**text_input)

                    elif isinstance(text_input, tuple) or isinstance(text_input, list):
                        if isinstance(text_input, tuple):
                            text_input = list(text_input)
                        for k in range(len(text_input)):
                            if isinstance(text_input[k], torch.Tensor):
                                text_input[k] = text_input[k].to(device)

                        text_vec = self.text_encoder(*text_input)

                    text_vecs.append(text_vec.detach().cpu())

        if img_vecs:
            img_vecs = torch.cat(img_vecs, dim=0).detach().numpy()
        else:
            img_vecs = None
        if text_vecs:
            text_vecs = torch.cat(text_vecs, dim=0).detach().numpy()
        else:
            text_vecs = None

        np.save("mscoco_img", img_vecs)
        np.save("mscoco_text", text_vecs)

        return img_vecs, text_vecs

    def _add_to_index(self, index, data):
        """Add `data` to faiss `index`. Because we want to use cosine similarity as our measure,
        we first normalize the vectors.

        Args:
            index (faiss.Index): Faiss index to which we are adding `data`.
            data (np.ndarray): Numpy 2D array of N x D where N is number of samples and D is dimension of vector representation of each sample.
        """
        faiss.normalize_L2(data)
        index.add(data)

    def build(self, dataset, batch_size=32, collate_fn=None, device="cpu"):
        """Build the index of `dataset`
        Args:
            dataset (torch.utils.data.Dataset): PyTorch map-style object that returns tuple of `(id, image_tensor, token_tensor)`
            or `(id, image_tensor)` or `(id, token_tensor)`. Image tensor is expected to be 2D and token tensor is expected to be 1-D.
                when elements are accessed using `__getitem__`.
            batch_size (int): Batch size for processing
            collate_fn (Callable): function used for creating a batch of samples.
            device (str|torch.device): Torch device to run the models. It can either be a string specifiying the device (e.g. "cuda:1")
                or the actual device object. Default is "cpu".
            parallel (bool): If True, wrap the models in `torch.nn.DataParallel` and run in parallel.
        """
        logger.info(f"Building joint embedding index.")
        if isinstance(device, str):
            device = torch.device(device)

        img_vecs, text_vecs = self._process_data(
            dataset, batch_size, collate_fn, device
        )

        if img_vecs is not None:
            self._add_to_index(self._img_index, img_vecs)

        if text_vecs is not None:
            self._add_to_index(self._text_index, text_vecs)

        logger.info(f"Finished building index.")

    def query(
        self, query, top_k=20, query_type="text", index_type="both", reduction="mean"
    ):
        """
        Query index with `query`
        Args:
            query (Union[str|PIL.Image.Image']): Query could either be a string (e.g. for retreiving image using text queries)
                or a PIL image (e.g. for retreiving relevant captions for a given image).
            top_k (int): The top-K results to return.
            query_type (str): The type of query. Options are "text" or "image".
            index_type (str): The type of index to perform lookup. Options are "both", "image", and "text".
                "both" attempts to look up both image and text indices and combine the results via `reduction` method.
                "image" only looks at the image index, while "text" only looks at the text index.
            reduction (str): The type of reduction to perform when `index_type == "both"` (options: "mean", "sum", "max").
                "mean" takes the score of image index and text index and average them. "sum" performs a simple summation.
                "max" takes the largest one as the score for the particular image/document.
        Returns:
            results (list[str|int]): List of global unique IDs of the top-K documents.
        """
        if query_type not in {"text", "image"}:
            raise ValueError(
                '`query_type` must be one of the following: `["text", "image"]'
            )
        if index_type not in {"both", "image", "text"}:
            raise ValueError(
                '`index_type` must be one of the following: `["both", "image", "text"]'
            )
        if reduction not in {"mean", "sum", "max"}:
            raise ValueError(
                '`reduction` must be one of the following: `["mean", "sum", "max"]'
            )
        if not isinstance(top_k, int):
            raise ValueError("`top_k` parameter must be an integer")

        if index_type == "both" and not (
            self._img_index.ntotal > 0 and self._text_index.ntotal > 0
        ):
            return ValueError(
                'Cannot run query with `index_type=="both"` when image and text indices are empty'
            )
        if index_type == "image" and not self._img_index.ntotal > 0:
            return ValueError(
                'Cannot run query with `index_type=="image"` when image index is empty.'
            )
        if index_type == "text" and not self._text_index.ntotal > 0:
            return ValueError(
                'Cannot run query with `index_type=="text"` when image index is empty.'
            )

        if query_type == "image":
            if isinstance(query, PIL.Image.Image):
                image = self.transformation(query)
                with torch.no_grad():
                    query_vec = self.img_encoder(image).detach().cpu().numpy()
            else:
                raise TypeError(
                    f'Type mismatch: `query_type=="image"` but `query` is of type {type(query)}.'
                )
        elif query_type == "text":
            if isinstance(query, str):
                text_tokens = self.tokenizer(query)
                length = len(text_tokens)
                text_tokens = torch.tensor(text_tokens).unsqueeze(0)
                with torch.no_grad():
                    query_vec = (
                        self.text_encoder(text_tokens, [length]).detach().cpu().numpy()
                    )
            else:
                raise TypeError(
                    f'Type mismatch: `query_type=="text"` but `query` is of type {type(query)}.'
                )
        else:
            raise ValueError(
                "`query` must either be a string or instance of `PIL.Image.Image` class."
            )

        # normalize the vector
        faiss.normalize_L2(query_vec)

        if index_type == "both":
            img_dist, img_indices = self._img_index.search(query_vec, top_k)
            text_dist, text_indices = self._text_index.search(query_vec, top_k)

            # Apply reduction
            combined_scores = {}
            for score, idx in zip(img_dist[0], img_indices[0]):
                combined_scores[idx] = score
            for score, idx in zip(text_dist[0], text_indices[0]):
                if idx in combined_scores:
                    if reduction == "mean":
                        combined_scores[idx] = (combined_scores[idx] + score) / 2
                    elif reduction == "sum":
                        combined_scores[idx] += score
                    elif reduction == "max":
                        combined_scores[idx] = max(combined_scores[idx], score)

            indices = sorted(
                combined_scores, key=lambda k: combined_scores[k], reverse=True
            )[:top_k]

        elif index_type == "image":
            dist, indices = self._img_index.search(query_vec, top_k)
            indices = indices[0]
        else:
            dist, indices = self._text_index.search(query_vec, top_k)
            indices = indices[0]

        # TODO check if sorting by dist is required

        return [self._idx2guid[i] for i in indices]

    def save(self, path):
        """Save index to directory. This involves saving the Faiss indices as
        well as id mappings.

        Args:
            path (str): Path of directory to save the contents of this index.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        logger.info(f"Saving index at {path}")

        img_index_path = os.path.join(path, "img.index")
        faiss.write_index(self._img_index, img_index_path)

        text_index_path = os.path.join(path, "text.index")
        faiss.write_index(self._text_index, text_index_path)

        id_mapping = (self._guid2idx, self._idx2guid)
        id_mapping_path = os.path.join(path, "mappings.pkl")
        with open(id_mapping_path, "wb") as f:
            pickle.dump(id_mapping, f)

    def load(self, path):
        """Load Faiss indices and id mappings.

        Args:
            path (str): Path of directory where the contents of this index is stored.
        """
        assert os.path.exists(path)

        logger.info(f"Loading index from {path}")

        id_mapping_path = os.path.join(path, "mappings.pkl")
        with open(id_mapping_path, "rb") as f:
            self._guid2idx, self._idx2guid = pickle.load(f)

        img_index_path = os.path.join(path, "img.index")
        assert os.path.exists(img_index_path)
        self._img_index = faiss.read_index(img_index_path)

        text_index_path = os.path.join(path, "text.index")
        assert os.path.exists(text_index_path)
        self._text_index = faiss.read_index(text_index_path)


class JointEmbeddingTfidfIndex(BaseIndex):
    """
    Index for the joint embedding space of images and text.
    It is composed of two faiss indices: (1) image index, (2) text index.
    Image index contains the vector representations of images, while text index contains the vector representation of text/captions.
    The idea is that given a query (e.g. string), we query both indices and combine the results.
    Similarity is measured using cosine similarity of the vectors.

    Args:
        dim (int): The dimension size of the faiss index.
        img_encoder (torch.nn.Module): The model for encoding image into a dense vector representation.
        transformation (torchvision.Transforms): The transformation steps applied to PIL.Image.Image objects to turn them into tensors.
        text_encoder (torch.nn.Module): The model for encoding text into a dense vector representation.
        tokenizer (obj): Tokenizer that when called (via `__call__`), returns list of ids that could be converted into a tensor and passed
            to the `text_encoder`.
    """

    def __init__(self, dim, img_encoder, transformation, text_encoder, tokenizer):
        self.dim = dim
        self.img_encoder = img_encoder
        self.transformation = transformation
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

        self._guid2idx = {}
        self._idx2guid = []
        self._img_index = faiss.IndexFlatIP(self.dim)
        self._text_index = faiss.IndexFlatIP(self.dim)

        self.img_encoder.eval()
        self.text_encoder.eval()

    def _process_data(self, dataset, batch_size, collate_fn, device):
        if not isinstance(dataset, torch.utils.data.Dataset):
            raise ValueError("`dataset` must be of type `torch.utils.data.Dataset`.")

        if len(dataset) <= 0:
            raise ValueError("`dataset` must have size greater than 0.")

        self.img_encoder.to(device)
        self.text_encoder.to(device)

        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn
        )
        i = 0
        img_vecs = []
        text_vecs = []

        logger.info(f"Processing dataset of size {len(dataset)}.")
        with torch.no_grad():
            for batch in tqdm(dataloader):
                # Run basic checks
                if len(batch) < 2:
                    raise ValueError(
                        "Tuple returned by `dataset` must contain at least two elements."
                    )
                if not (isinstance(batch[0][0], int) or isinstance(batch[0][0], str)):
                    raise ValueError(
                        "First element of tuple returned by `dataset` must be an integer or a string (ID must be hashable)"
                    )

                ids = batch[0]
                for _id in ids:
                    self._guid2idx[_id] = i
                    self._idx2guid.append(_id)
                    i += 1

                text_input = None
                if isinstance(batch[1], torch.Tensor) and len(batch[1][0].shape) >= 2:
                    # `batch[1]` is an image tensor
                    img = batch[1].to(device)
                    img_vec = self.img_encoder(img)
                    img_vecs.append(img_vec.detach().cpu())
                else:
                    # `batch[1]` is an token tensor
                    text_input = batch[1]

                if len(batch) == 3:
                    # If `batch` has three elements, then third element is token tensor
                    text_input = batch[2]

                if text_input is not None:
                    # Handle different formats of `text_input`
                    if isinstance(text_input, dict):
                        for k in text_input:
                            if isinstance(text_input[k], torch.Tensor):
                                text_input[k] = text_input[k].to(device)

                        text_vec = self.text_encoder(**text_input)

                    elif isinstance(text_input, tuple) or isinstance(text_input, list):
                        if isinstance(text_input, tuple):
                            text_input = list(text_input)
                        for k in range(len(text_input)):
                            if isinstance(text_input[k], torch.Tensor):
                                text_input[k] = text_input[k].to(device)

                        text_vec = self.text_encoder(*text_input)

                    text_vecs.append(text_vec.detach().cpu())

        if img_vecs:
            img_vecs = torch.cat(img_vecs, dim=0).detach().numpy()
        else:
            img_vecs = None
        if text_vecs:
            text_vecs = torch.cat(text_vecs, dim=0).detach().numpy()
        else:
            text_vecs = None

        np.save("mscoco_img", img_vecs)
        np.save("mscoco_text", text_vecs)

        return img_vecs, text_vecs

    def _add_to_index(self, index, data):
        """Add `data` to faiss `index`. Because we want to use cosine similarity as our measure,
        we first normalize the vectors.

        Args:
            index (faiss.Index): Faiss index to which we are adding `data`.
            data (np.ndarray): Numpy 2D array of N x D where N is number of samples and D is dimension of vector representation of each sample.
        """
        faiss.normalize_L2(data)
        index.add(data)

    def build(self, dataset, batch_size=32, collate_fn=None, device="cpu"):
        """Build the index of `dataset`
        Args:
            dataset (torch.utils.data.Dataset): PyTorch map-style object that returns tuple of `(id, image_tensor, token_tensor)`
            or `(id, image_tensor)` or `(id, token_tensor)`. Image tensor is expected to be 2D and token tensor is expected to be 1-D.
                when elements are accessed using `__getitem__`.
            batch_size (int): Batch size for processing
            collate_fn (Callable): function used for creating a batch of samples.
            device (str|torch.device): Torch device to run the models. It can either be a string specifiying the device (e.g. "cuda:1")
                or the actual device object. Default is "cpu".
            parallel (bool): If True, wrap the models in `torch.nn.DataParallel` and run in parallel.
        """
        logger.info(f"Building joint embedding index.")
        if isinstance(device, str):
            device = torch.device(device)

        img_vecs, text_vecs = self._process_data(
            dataset, batch_size, collate_fn, device
        )

        if img_vecs is not None:
            self._add_to_index(self._img_index, img_vecs)

        if text_vecs is not None:
            self._add_to_index(self._text_index, text_vecs)

        logger.info(f"Finished building index.")

    def query(
        self, query, top_k=20, query_type="text", index_type="both", reduction="mean"
    ):
        """
        Query index with `query`
        Args:
            query (Union[str|PIL.Image.Image']): Query could either be a string (e.g. for retreiving image using text queries)
                or a PIL image (e.g. for retreiving relevant captions for a given image).
            top_k (int): The top-K results to return.
            query_type (str): The type of query. Options are "text" or "image".
            index_type (str): The type of index to perform lookup. Options are "both", "image", and "text".
                "both" attempts to look up both image and text indices and combine the results via `reduction` method.
                "image" only looks at the image index, while "text" only looks at the text index.
            reduction (str): The type of reduction to perform when `index_type == "both"` (options: "mean", "sum", "max").
                "mean" takes the score of image index and text index and average them. "sum" performs a simple summation.
                "max" takes the largest one as the score for the particular image/document.
        Returns:
            results (list[str|int]): List of global unique IDs of the top-K documents.
        """
        if query_type not in {"text", "image"}:
            raise ValueError(
                '`query_type` must be one of the following: `["text", "image"]'
            )
        if index_type not in {"both", "image", "text"}:
            raise ValueError(
                '`index_type` must be one of the following: `["both", "image", "text"]'
            )
        if reduction not in {"mean", "sum", "max"}:
            raise ValueError(
                '`reduction` must be one of the following: `["mean", "sum", "max"]'
            )
        if not isinstance(top_k, int):
            raise ValueError("`top_k` parameter must be an integer")

        if index_type == "both" and not (
            self._img_index.ntotal > 0 and self._text_index.ntotal > 0
        ):
            return ValueError(
                'Cannot run query with `index_type=="both"` when image and text indices are empty'
            )
        if index_type == "image" and not self._img_index.ntotal > 0:
            return ValueError(
                'Cannot run query with `index_type=="image"` when image index is empty.'
            )
        if index_type == "text" and not self._text_index.ntotal > 0:
            return ValueError(
                'Cannot run query with `index_type=="text"` when image index is empty.'
            )

        if query_type == "image":
            if isinstance(query, PIL.Image.Image):
                image = self.transformation(query)
                with torch.no_grad():
                    query_vec = self.img_encoder(image).detach().cpu().numpy()
            else:
                raise TypeError(
                    f'Type mismatch: `query_type=="image"` but `query` is of type {type(query)}.'
                )
        elif query_type == "text":
            if isinstance(query, str):
                text_tokens = self.tokenizer(query)
                length = len(text_tokens)
                text_tokens = torch.tensor(text_tokens).unsqueeze(0)
                with torch.no_grad():
                    query_vec = (
                        self.text_encoder(text_tokens, [length]).detach().cpu().numpy()
                    )
            else:
                raise TypeError(
                    f'Type mismatch: `query_type=="text"` but `query` is of type {type(query)}.'
                )
        else:
            raise ValueError(
                "`query` must either be a string or instance of `PIL.Image.Image` class."
            )

        # normalize the vector
        faiss.normalize_L2(query_vec)

        if index_type == "both":
            img_dist, img_indices = self._img_index.search(query_vec, top_k)
            text_dist, text_indices = self._text_index.search(query_vec, top_k)

            # Apply reduction
            combined_scores = {}
            for score, idx in zip(img_dist[0], img_indices[0]):
                combined_scores[idx] = score
            for score, idx in zip(text_dist[0], text_indices[0]):
                if idx in combined_scores:
                    if reduction == "mean":
                        combined_scores[idx] = (combined_scores[idx] + score) / 2
                    elif reduction == "sum":
                        combined_scores[idx] += score
                    elif reduction == "max":
                        combined_scores[idx] = max(combined_scores[idx], score)

            indices = sorted(
                combined_scores, key=lambda k: combined_scores[k], reverse=True
            )[:top_k]

        elif index_type == "image":
            dist, indices = self._img_index.search(query_vec, top_k)
            indices = indices[0]
        else:
            dist, indices = self._text_index.search(query_vec, top_k)
            indices = indices[0]

        # TODO check if sorting by dist is required

        return [self._idx2guid[i] for i in indices]

    def save(self, path):
        """Save index to directory. This involves saving the Faiss indices as
        well as id mappings.

        Args:
            path (str): Path of directory to save the contents of this index.
        """
        if not os.path.exists(path):
            os.makedirs(path)

        logger.info(f"Saving index at {path}")

        img_index_path = os.path.join(path, "img.index")
        faiss.write_index(self._img_index, img_index_path)

        text_index_path = os.path.join(path, "text.index")
        faiss.write_index(self._text_index, text_index_path)

        id_mapping = (self._guid2idx, self._idx2guid)
        id_mapping_path = os.path.join(path, "mappings.pkl")
        with open(id_mapping_path, "wb") as f:
            pickle.dump(id_mapping, f)

    def load(self, path):
        """Load Faiss indices and id mappings.

        Args:
            path (str): Path of directory where the contents of this index is stored.
        """
        assert os.path.exists(path)

        logger.info(f"Loading index from {path}")

        id_mapping_path = os.path.join(path, "mappings.pkl")
        with open(id_mapping_path, "rb") as f:
            self._guid2idx, self._idx2guid = pickle.load(f)

        img_index_path = os.path.join(path, "img.index")
        assert os.path.exists(img_index_path)
        self._img_index = faiss.read_index(img_index_path)

        text_index_path = os.path.join(path, "text.index")
        assert os.path.exists(text_index_path)
        self._text_index = faiss.read_index(text_index_path)
