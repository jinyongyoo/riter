# Demo webapp for Riter
This is a simple flask app that uses Riter to search images from COCO and Flickr30k datasets. 

## Installation
First, run `pip install -r requirements.txt` to install the required dependencies.

## Building Indices
We first need to build `riter.JointEmbeddingIndex` indices for COCO and Flickr30k datasets. 

### COCO Dataset
First, go to [COCO webste](https://cocodataset.org/#home) and download 2017 train and validation images and annotations. Also, you can run `wget` on these following links and unzip them.
- 2017 train images (18GB): http://images.cocodataset.org/zips/train2017.zip
- 2017 validation images (1GB): http://images.cocodataset.org/zips/val2017.zip
- 2017 train & val annotations: http://images.cocodataset.org/annotations/annotations_trainval2017.zip

### Flickr30k Dataset
TBD

Next, please run `python scripts/build_coco_index.py` and `python scripts/build_flickr_index.py` to build these indices. For both scripts, you need to provide appropriate paths to the images, annotation JSON file. We recommend that you use a GPU while building these indices. If you do not specify a specific path to save your index, they should appear under `./saved` directory.

## Running the app locally
Run `export FLASH_APP=app.py`. Then, run `flask run`. You can checkout the web app by typing in `localhost:9999`
## Bugs
If you are encountering a  Error #15: Initializing libiomp5.dylib, but found libomp.dylib already initialized. issue when doing python app.py
```
 os.environ['KMP_DUPLICATE_LIB_OK']='True'
```
