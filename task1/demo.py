import os
import sys
import cv2
import time
import copy
import torch
import argparse
import numpy as np
import os.path as osp

sys.path.append(os.getcwd())

from PIL import Image
from vizer.draw import draw_boxes
from task1.model import CTPN
from task1.configs import configs
from task1.utils.dset import read_image
from task1.utils.logger import create_logger
from task1.data.postprocessing import TextDetector, remove_empty_boxes
from task1.data.preprocessing.augmentation import BasicDataTransformation
from task1.data.preprocessing.transformations import ToSobelGradient, ToMorphology, \
    CropImage, ConvertColor

def get_args ():
    parser = argparse.ArgumentParser(description="CTPN: prediction phase")
    parser.add_argument("--config-file", action="store", help="The path to the configs file.")
    parser.add_argument("--trained-model", action="store", help="The path to the trained model state dict file")
    parser.add_argument("--image-folder", default="demo/images", type=str,
                        help="The path to the trained model state dict file")
    parser.add_argument("--output-folder", default="demo/results/ctpn", type=str,
                        help="The directory to save prediction results")
    parser.add_argument("--remove-extra-white", action="store_true",
                        help="Enable or disable the need for removing the extra white space on the scanned receipts."
                            "By default it is disable.")
    parser.add_argument("--use-cuda", action="store_true",
                        help="enable/disable cuda during prediction. By default it is disable")
    parser.add_argument("--use-amp", action="store_true",
                        help="Enable or disable the automatic mixed precision. By default it is disable."
                            "For further info, check those following links:"
                            "https://pytorch.org/docs/stable/amp.html"
                            "https://pytorch.org/docs/stable/notes/amp_examples.html"
                            "https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html")
    parser.add_argument("--gpu-device", default=0, type=int, help="Specify the GPU ID to use. By default, it is the ID 0")
    return parser.parse_args()

class OneImagePrediction:

    def __init__(self, model, remove_extra_white = False, \
        use_cuda = False, use_amp = False, gpu_device = 0):

        self.remove_extra_white = remove_extra_white

        # A boolean to check whether the user is able to use cuda or not.
        use_cuda = torch.cuda.is_available() and use_cuda

        # A boolean to check whether the user is able to use amp or not.
        self.use_amp = use_amp and use_cuda

        # The declaration and tensor type of the CPU/GPU device.
        if not use_cuda:
            self.device = torch.device("cpu")
            torch.set_default_tensor_type('torch.FloatTensor')
        else:
            self.device = torch.device("cuda")
            torch.cuda.set_device(gpu_device)
            torch.set_default_tensor_type("torch.cuda.FloatTensor")

            torch.backends.cudnn.enabled = True

        self.model = model.to(self.device)
        self.text_detector = TextDetector(configs=configs)
        self.basic_transform = BasicDataTransformation(configs)

        if self.remove_extra_white:
            # A set of classes for removing extra white space.
            self.cropImage = CropImage()
            self.morphologyEx = ToMorphology()
            self.grayColor = ConvertColor(current="RGB", transform="GRAY")
            self.sobelGradient = ToSobelGradient(cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    @torch.no_grad()
    def run(self,image): # image is nparray, return the detected_bboxes, detected_scores
        self.model = self.model.eval()
        original_image = image

        # Apply the the same crop logic as in the preprocessing step.
        cropped_pixels_width = cropped_pixels_height = 0
        if self.remove_extra_white and image.shape[1] > 990:
            gray_image = self.grayColor(image)[0]
            threshed_image = self.sobelGradient(gray_image)
            morpho_image = self.morphologyEx(threshed_image)
            image, cropped_pixels_width, cropped_pixels_height = self.cropImage(morpho_image, image)

        original_image_height, original_image_width = image.shape[:2]

        image = self.basic_transform(image, None)[0]

        new_image_height, new_image_width = image.shape[1:]

        image = image.to(self.device)

        image = image.unsqueeze(0)  # Shape: [1, channels, height, width]

        # Forward pass using AMP if it is set to True.
        # autocast may be used by itself to wrap inference or evaluation forward passes.
        # GradScaler is not necessary.
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            predictions = self.model(image)

        # Detections
        detections = self.text_detector(predictions, image_size=(new_image_height, new_image_width))
        detected_bboxes, detected_scores = detections

        # Scaling the bounding boxes back to the original image.
        ratio_w = original_image_width / new_image_width
        ratio_h = original_image_height / new_image_height
        size_ = np.array([[ratio_w, ratio_h, ratio_w, ratio_h]])
        detected_bboxes *= size_

        # Adjusting the bounding box coordinates, if the images were previously cropped.
        detected_bboxes[:, 0::2] += cropped_pixels_width
        detected_bboxes[:, 1::2] += cropped_pixels_height

        # Removing empty bounding boxes.
        qualified_bbox_indices = remove_empty_boxes(original_image, detected_bboxes)
        detected_bboxes = detected_bboxes[qualified_bbox_indices]

        # I think the original author is wrong, so I try to write the next line
        detected_scores = detected_scores[qualified_bbox_indices]

        return detected_bboxes, detected_scores

def get_CTPN_model_default_config (trained_model_path):
    # Initialisation and loading the model's weight
    modelArgs = dict(configs.MODEL.ARGS)
    model = CTPN(**modelArgs)
    checkpoint = torch.load(trained_model_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

class Prediction:
    def __init__(self, trained_model_path, image_folder, output_folder,\
        remove_extra_white = False,
        use_cuda = False, use_amp = False, gpu_device = 0):

        self.output_folder = output_folder
        self.remove_extra_while = remove_extra_white

        possible_extension_image = ("jpg", "png", "jpeg", "JPG")

        files = list(sorted(os.scandir(path=image_folder), key=lambda f: f.name))
        self.images = [f for f in files if f.name.endswith(possible_extension_image)]

        if len(self.images) == 0:
            raise ValueError("There are no images for prediction!")

        if not osp.exists(output_folder):
            os.makedirs(output_folder)

        output_dir = os.path.normpath(configs.OUTPUT_DIR)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        self.logger = create_logger(name="Prediction phase", output_dir=output_dir, log_filename="prediction-log")
        self.model = get_CTPN_model_default_config(trained_model_path)
        self.predictor = OneImagePrediction(self.model, remove_extra_white, use_cuda, use_amp, gpu_device)

    @torch.no_grad()
    def run(self):

        self.model = self.model.eval()

        for image in self.images:

            start = time.time()
            original_image_path = image.path
            original_image = np.array(read_image(original_image_path))
            detected_bboxes, detected_scores = self.predictor.run(original_image)
            prediction_time = time.time()-start

            self.logger.info("Image : {image} || "
                            "Prediction time: {pt:.3f} ms || "
                            "Objects detected: {objects}\n".format(
                image = image.path,
                pt=round(prediction_time * 1000),
                objects = len(detected_bboxes)
            ))

            # Drawing the bounding boxes on the original image.
            drawn_image = draw_boxes(image=original_image, boxes=detected_bboxes)

            # Saving the drawn image.
            image_name, image_ext = os.path.splitext(os.path.basename(original_image_path))
            Image.fromarray(drawn_image).save(os.path.join(self.output_folder, image_name + image_ext))

            # Writing the annotation .txt file
            with open(os.path.join(self.output_folder, image_name + ".txt"), "w") as f:
                for j, coords in enumerate(detected_bboxes):
                    line = ",".join(str(round(coord)) for coord in coords)
                    line += ", {0}".format(detected_scores[j])
                    line += "\n"
                    f.write(line)


if __name__ == '__main__':
    args = get_args()
    # Guarding against bad arguments.

    if args.trained_model is None:
        raise ValueError("The path to the trained model is not provided!")
    elif not osp.isfile(args.trained_model):
        raise ValueError("The path to the trained model is wrong!")

    gpu_devices = list(range(torch.cuda.device_count()))
    if len(gpu_devices) != 0 and args.gpu_device not in gpu_devices:
        raise ValueError("Your GPU ID is out of the range! You may want to check with 'nvidia-smi'")
    elif args.use_cuda:
        if not torch.cuda.is_available():
            raise ValueError("The argument --use-cuda is specified but it seems you cannot use CUDA!")
    elif args.use_amp:
        raise ValueError("The arguments --use-cuda, --use-amp and --gpu-device must be used together!")

    if args.config_file is not None:
        if not os.path.isfile(args.config_file):
            raise ValueError("The configs file is wrong!")
        else:
            configs.merge_from_file(args.config_file)
    configs.freeze()

    Prediction(args.trained_model, args.image_folder, args.output_folder,
        args.remove_extra_white, args.use_cuda, args.use_amp, args.gpu_device).run()
