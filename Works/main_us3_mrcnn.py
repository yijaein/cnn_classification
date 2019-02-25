"""
Mask R-CNN
Train on the nuclei segmentation dataset from the
Kaggle 2018 Data Science Bowl
https://www.kaggle.com/c/data-science-bowl-2018/

Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from ImageNet weights
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=imagenet

    # Train a new model starting from specific weights file
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=/path/to/weights.h5

    # Resume training a model that you had trained earlier
    python3 nucleus.py train --dataset=/path/to/dataset --subset=train --weights=last

    # Generate submission file
    python3 nucleus.py detect --dataset=/path/to/dataset --subset=train --weights=<last or /path/to/weights.h5>
"""

# Set matplotlib backend
# This has to be done before other importa that might
# set it, but only if we're running in script mode
# rather than being imported.
if __name__ == '__main__':
    import matplotlib

    # Agg backend runs without a display
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

import datetime
import glob
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
from imgaug import augmenters as iaa

from Tools.crop_by_seg import calculate_angle, rotate_bound, find_bounding_square

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib
from mrcnn import visualize

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

# Results directory
# Save submission files here
RESULTS_DIR = os.path.join(ROOT_DIR, "results_us3_mrcnn_all/400")


############################################################
#  Configurations
############################################################

class KidneyConfig(Config):
    """Configuration for training on the nucleus segmentation dataset."""
    # Give the configuration a recognizable name
    NAME = "kidney_oriUS"

    GPU_COUNT = 2

    # Adjust depending on your GPU memory
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + nucleus

    # Number of training and validation steps per epoch
    STEPS_PER_EPOCH = 1000  # 12513 // IMAGES_PER_GPU
    VALIDATION_STEPS = 50  # max(1, len(VAL_IMAGE_IDS) // IMAGES_PER_GPU)

    # Don't exclude based on confidence. Since we have two classes
    # then 0.5 is the minimum anyway as it picks between nucleus and BG
    DETECTION_MIN_CONFIDENCE = 0.9

    DETECTION_MAX_INSTANCES = 1
    MAX_GT_INSTANCES = 1

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Input image resizing
    # Random crops of size 512x512
    # IMAGE_RESIZE_MODE = "crop"
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 512
    # IMAGE_MIN_SCALE = 2.0

    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 1000
    POST_NMS_ROIS_INFERENCE = 2000

    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.9

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 64

    # Image mean (RGB)
    MEAN_PIXEL = np.array([43.53, 39.56, 48.22])

    # If enabled, resizes instance masks to a smaller size to reduce
    # memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False
    MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask

    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    TRAIN_ROIS_PER_IMAGE = 128

    # Maximum number of ground truth instances to use in one image
    MAX_GT_INSTANCES = 200

    # Max number of final detections per image
    DETECTION_MAX_INSTANCES = 400


class KidneyInferenceConfig(KidneyConfig):
    # Set batch size to 1 to run one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    # Don't resize imager for inferencing
    IMAGE_RESIZE_MODE = "pad64"
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more propsals.
    RPN_NMS_THRESHOLD = 0.7


############################################################
#  Dataset
############################################################

class UltrasoundDataset(utils.Dataset):

    def load_kidney(self, dataset_dir, subset, seg_dir):
        """Load a subset of the nuclei dataset.

        dataset_dir: Root directory of the dataset
        subset: Subset to load. Either the name of the sub-directory,
                such as stage1_train, stage1_test, ...etc. or, one of:
                * train: stage1_train excluding validation images
                * val: validation images from VAL_IMAGE_IDS
        """
        # Add classes. We have one class.
        # Naming the dataset nucleus, and the class nucleus
        self.add_class("kidney", 1, "kidney")

        # Which subset?
        # "val": use hard-coded list above
        # "train": use data from stage1_train minus the hard-coded list above
        # else: use the data from the specified sub-directory
        assert subset in ["train", "val", ""]
        if subset != "":
            dataset_subset_dir = os.path.join(dataset_dir, subset)
        else:
            dataset_subset_dir = dataset_dir

        image_files = glob.glob(os.path.join(dataset_subset_dir, '**', '*.png'), recursive=True)

        # Add images
        for file in image_files:
            _, name_ext = os.path.split(file)
            name = os.path.splitext(name_ext)[0]

            height, width = cv2.imread(file, cv2.IMREAD_GRAYSCALE).shape[:2]
            self.add_image(
                source="kidney",
                image_id=name,
                path=file,
                height=height,
                width=width)

        self.mask_dict = dict()
        for (root, dirs, files) in os.walk(seg_dir):
            for file in files:
                name, ext = os.path.splitext(file)
                if ext.lower() not in ['.png', '.jpg']:
                    continue
                self.mask_dict[name] = os.path.join(root, file)
        print('cnt mask', len(self.mask_dict.keys()))
        print('cnt img', len(self.image_info))

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        info = self.image_info[image_id]
        id = info['id']

        mask = []
        # Read mask files from .png image
        if id in self.mask_dict.keys():
            mask_file = self.mask_dict[id]
            m = skimage.io.imread(mask_file).astype(np.bool)
            class_ids = np.ones([1, ], dtype=np.int32)
            mask.append(m)
        else:
            m = np.zeros((info['height'], info['width']), dtype=np.bool)
            class_ids = np.zeros([1, ], dtype=np.int32)
            mask.append(m)

        mask = np.stack(mask, axis=-1)
        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID, we return an array of ones

        return mask, class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "kidney":
            return info["id"]
        else:
            super(self.__class__, self).image_reference(image_id)


############################################################
#  Training
############################################################

def train(model, dataset_dir, seg_dir):
    """Train the model."""
    # Training dataset.
    dataset_train = UltrasoundDataset()
    dataset_train.load_kidney(dataset_dir, "train", seg_dir)
    dataset_train.prepare()

    # Validation dataset
    dataset_val = UltrasoundDataset()
    dataset_val.load_kidney(dataset_dir, "val", seg_dir)
    dataset_val.prepare()

    # Image augmentation
    # http://imgaug.readthedocs.io/en/latest/source/augmenters.html
    augmentation = iaa.SomeOf((0, 6), [
        iaa.Fliplr(0.5),
        iaa.Multiply((0.8, 1.5)),
        iaa.GaussianBlur(sigma=(0.0, 5.0)),
        iaa.Add((-10, 10), per_channel=0.5),
        iaa.PiecewiseAffine(scale=(0.01, 0.02)),
        iaa.Affine(rotate=(-10, 10))
    ])

    # *** This training schedule is an example. Update to your needs ***

    # If starting from imagenet, train heads only for a bit
    # since they have random weights
    print("Train network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=5,
                augmentation=augmentation,
                layers='heads')

    print("Train all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=200,
                augmentation=augmentation,
                layers='all')


############################################################
#  Detection
############################################################

def detect(model, dataset_dir, subset, seg_dir):
    """Run detection on images in the given directory."""
    print("Running on {}".format(dataset_dir))

    # Create directory
    def exists_path(p):
        if not os.path.exists(p):
            os.makedirs(p)
        return p

    result_dir = exists_path(os.path.join(RESULTS_DIR, "{:%Y%m%dT%H%M%S}".format(datetime.datetime.now())))
    result_seg_dir = exists_path(os.path.join(result_dir, 'SegKidney_MRCNN'))
    result_crop_dir = exists_path(os.path.join(result_dir, 'CrioKidney_MRCNN'))

    # Read dataset
    dataset = UltrasoundDataset()
    dataset.load_kidney(dataset_dir, subset, seg_dir)
    dataset.prepare()

    # Load over images
    for image_id in dataset.image_ids:
        # Load image and run detection
        image = dataset.load_image(image_id)
        # Detect objects
        r = model.detect([image], verbose=0)[0]
        # Encode image to RLE. Returns a string of multiple lines
        source_id = dataset.image_info[image_id]["id"]

        for idx in range(r['masks'].shape[2]):
            # save SegKidney
            m = r['masks'][:, :, idx].astype(np.uint8) * 255
            cv2.imwrite(os.path.join(result_seg_dir, source_id + '#' + str(idx) + '.png'), m)

            # save CropKidney
            img_ori = cv2.imread(dataset.image_info[image_id]['path'], cv2.IMREAD_GRAYSCALE)
            crop_img = crop_by_mask(img_ori, m)
            cv2.imwrite(os.path.join(result_crop_dir, source_id + '.png'), crop_img)

        # Save image with masks
        visualize.display_instances(
            image, r['rois'], r['masks'], r['class_ids'],
            dataset.class_names, r['scores'],
            show_bbox=False, show_mask=False,
            title="Predictions")
        plt.savefig(os.path.join(result_dir, source_id + '.png'))
        plt.close('all')


def crop_by_mask(img_ori, img_mask, padding_size=20, seg_padding_size=0, rotate=True):
    # add pad to seg image
    if seg_padding_size > 0:
        _, contours, hierachy = cv2.findContours(img_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img_mask = cv2.drawContours(img_mask, contours, -1, 255, seg_padding_size)

    # rotate image
    if rotate:
        angle = calculate_angle(img_mask)
        img_mask = rotate_bound(img_mask, angle)
        img_ori = rotate_bound(img_ori, angle)

    # Add black padding to input image and seg image
    img_mask = cv2.copyMakeBorder(img_mask, padding_size, padding_size, padding_size, padding_size, 0)
    img_ori = cv2.copyMakeBorder(img_ori, padding_size, padding_size, padding_size, padding_size, 0)

    # get white pixel bounding box
    x, y, w, h = find_bounding_square(img_mask, padding=padding_size)

    img_crop = img_ori[int(y):int(y + h), int(x):int(x + w)]

    return img_crop


############################################################
#  Command Line
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Mask R-CNN for nuclei counting and segmentation')
    parser.add_argument("--command",
                        default='detect',
                        metavar="<command>",
                        help="'train' or 'detect'")
    parser.add_argument('--dataset',
                        default='/media/bong07/895GB/data/yonsei/png_classification/180718_KidneyUS_400_png',
                        metavar="/path/to/dataset/",
                        help='Root directory of the dataset')
    parser.add_argument('--weights',
                        default='/home/bong07/lib/robin_yonsei/logs/kidney_orius20181113T1721/mask_rcnn_kidney_orius_0021.h5',
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco' or 'imagenet' or 'last' or ''")
    parser.add_argument('--logs',
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--subset',
                        default='',
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    parser.add_argument('--seg_dir',
                        default='~/data/SegKidney',
                        metavar="Dataset sub-directory",
                        help="Subset of dataset to run prediction on")
    args = parser.parse_args()

    args.dataset = os.path.expanduser(args.dataset)
    args.seg_dir = os.path.expanduser(args.seg_dir)

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    # elif args.command == "detect":
    #     assert args.subset, "Provide --subset to run prediction on"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    if args.subset:
        print("Subset: ", args.subset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = KidneyConfig()
    else:
        config = KidneyInferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    if args.weights != '':
        print("Loading weights ", weights_path)
        if args.weights.lower() == "coco":
            # Exclude the last layers because they require a matching
            # number of classes
            model.load_weights(weights_path, by_name=True, exclude=[
                "mrcnn_class_logits", "mrcnn_bbox_fc",
                "mrcnn_bbox", "mrcnn_mask"])
        else:
            model.load_weights(weights_path, by_name=True)
    else:
        print("Init weights")

    # Train or evaluate
    if args.command == "train":
        train(model, args.dataset, args.seg_dir)
    elif args.command == "detect":
        detect(model, args.dataset, args.subset, args.seg_dir)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'detect'".format(args.command))
