import os
from pycocotools.coco import COCO


class CocoDataset:
    """COCO Custom Dataset

    Args:
        img_root: path of image directory
        annotation_json: path of JSON file for COCO annotations
    """

    def __init__(self, img_root, annotation_json):
        self.img_root = img_root
        self.coco = COCO(annotation_json)

    def __getitem__(self, _id):
        img_id = self.coco.anns[_id]["image_id"]
        img_file = os.path.join(
            self.img_root, self.coco.loadImgs(img_id)[0]["file_name"]
        )

        text = self.coco.anns[_id]["caption"]

        return img_file, text
