import os
import pickle
import argparse
from functools import partial

import torch
import torchvision
from pycocotools.coco import COCO
from PIL import Image

from riter import JointEmbeddingIndex
from riter.prebuilt import Tokenizer, ImageEncoder, TextEncoder
from riter.prebuilt.config import COCO_GRU_CONFIG, COCO_RESNET_CONFIG


class CocoDataset(torch.utils.data.Dataset):
    """COCO Custom Dataset compatible with torch.utils.data.DataLoader.

    Args:
        img_root: path of image directory
        transform: image preprocessor
        annotation_json: path of JSON file for COCO annotations
        tokenizer: tokenizer for annotations
    """

    def __init__(self, img_root, transform, annotation_json, tokenizer):
        self.img_root = img_root
        self.coco = COCO(annotation_json)
        self.ids = list(self.coco.anns.keys())
        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        _id = self.ids[index]
        img_id = self.coco.anns[_id]["image_id"]
        img_file = self.coco.loadImgs(img_id)[0]["file_name"]

        image = Image.open(os.path.join(self.img_root, img_file)).convert("RGB")
        image = self.transform(image)

        text = self.coco.anns[_id]["caption"]
        tokens = torch.tensor(self.tokenizer(text))

        return _id, image, tokens

    def __len__(self):
        return len(self.ids)


def collate_fn(pad_token_id, data):
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[2]), reverse=True)
    ids, images, captions = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    captions = torch.nn.utils.rnn.pad_sequence(
        captions, batch_first=True, padding_value=pad_token_id
    )

    return list(ids), images, (captions, lengths)


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
        "--prebuilt-path",
        default="./saved/mscoco",
        type=str,
        help="Path to pretrained encoders and vocabulary",
    )
    parser.add_argument(
        "--img-encoder-name",
        default="resnet",
        type=str,
        help='Name of image encoder model (options: "resnet" and "vgg")',
    )
    parser.add_argument(
        "--batch-size", default=512, type=int, help="Batch size of processing data"
    )
    parser.add_argument(
        "--dim", default=1024, type=int, help="Dimension of joint-embedding"
    )
    parser.add_argument(
        "--index-save-path",
        default="./saved/mscoco/val_index",
        type=str,
        help="Path to save index",
    )
    args = parser.parse_args()

    tokenizer_path = os.path.join(args.prebuilt_path, "coco_tokenizer.pkl")
    tokenizer = Tokenizer.load(tokenizer_path)
    transform = ImageEncoder.TRANSFORM

    dataset = CocoDataset(
        args.img_data_path,
        transform,
        args.annotation_data_path,
        tokenizer,
    )

    collate_fn = partial(collate_fn, tokenizer.word2idx(Tokenizer.PAD_TOKEN))

    img_encoder = ImageEncoder(**COCO_RESNET_CONFIG)
    img_encoder.load_state_dict(
        torch.load(os.path.join(args.prebuilt_path, "coco_resnet.pth"))
    )
    text_encoder = TextEncoder(**COCO_GRU_CONFIG)
    text_encoder.load_state_dict(
        torch.load(os.path.join(args.prebuilt_path, "coco_gru.pth"))
    )

    index = JointEmbeddingIndex(
        args.dim, img_encoder, transform, text_encoder, tokenizer
    )
    index.build(
        dataset, batch_size=args.batch_size, collate_fn=collate_fn, device="cuda"
    )
    index.save(args.index_save_path)
