import os
import pickle
import argparse
from functools import partial

import torch
import torchvision
from pycocotools.coco import COCO
from PIL import Image

from riter import JointEmbeddingIndex
from riter import AutoModel, AutoTokenizer, AutoTransformation


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
        # For each image, there are multiple annotations. We'll justp pick the first pair.
        annotation_ids = list(self.coco.anns.keys())
        self.img_ids = set()
        self.annotation_ids = {}
        for i in annotation_ids:
            if self.coco.anns[i]["image_id"] not in self.img_ids:
                self.img_ids.add(self.coco.anns[i]["image_id"])
                self.annotation_ids[self.coco.anns[i]["image_id"]] = []
            self.annotation_ids[self.coco.anns[i]["image_id"]].append(i)
        self.img_ids = list(self.img_ids)
        self.img_ids.sort()
        self.transform = transform
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        img_file = self.coco.loadImgs(img_id)[0]["file_name"]

        image = Image.open(os.path.join(self.img_root, img_file)).convert("RGB")
        image = self.transform(image)

        text = " ".join(
            [self.coco.anns[i]["caption"] for i in self.annotation_ids[img_id]]
        )
        tokens = torch.tensor(self.tokenizer(text))

        return img_id, image, tokens

    def __len__(self):
        return len(self.img_ids)


def collate_fn(pad_token_id, data):
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[2]), reverse=True)
    ids, images, captions = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merge captions (convert tuple of 1D tensor to 2D tensor)
    lengths = torch.tensor([len(cap) for cap in captions], dtype=torch.int64)
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

    img_encoder = AutoModel.from_pretrained("vsepp-resnet-coco")
    transformation = AutoTransformation.from_pretrained("vsepp-resnet-coco")
    text_encoder = AutoModel.from_pretrained("vsepp-gru-coco")
    tokenizer = AutoTokenizer.from_pretrained("vsepp-gru-coco")

    index = JointEmbeddingIndex(
        args.dim, img_encoder, transformation, text_encoder, tokenizer
    )

    dataset = CocoDataset(
        args.img_data_path,
        transformation,
        args.annotation_data_path,
        tokenizer,
    )

    collate_fn = partial(collate_fn, tokenizer.pad_token_id)

    index.build(
        dataset, batch_size=args.batch_size, collate_fn=collate_fn, device="cuda"
    )
    index.save(args.index_save_path)
