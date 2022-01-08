import argparse
import torch
import os.path as osp
import sys
# This is a trick/hack to import demo
import task1.demo as task1
import task2.test as task2
import task3.demo as task3
import cv2
import json
from PIL import Image
import numpy as np
from vizer.draw import draw_boxes

"""
Task 1, task 2, task 3 if def main là chạy cả thư mục
Còn nếu từ main, nó chỉ cần hàm predict chính xác hình ảnh nào đó hay bounding box nào đó thôi
"""
# Task 1, task 2, task 3 cần hàm đưa model vào chạy đúng thứ mình cần

# Args task 1:
# Task 1 input 1 hình, output bboxes của hình đó.


def get_args():
    parser = argparse.ArgumentParser(description="Receipt reader")
    parser.add_argument("--detection-model-path", action="store", help="The path to the trained task1 model file. Required",required=True)
    parser.add_argument('--ocr-model-path',type=str,help='The path to the trained task2 model file. Required',required=True)
    parser.add_argument('--extraction-model-path',action="store",help='The path to the trained task 3 model file. Required.',required=True)

    parser.add_argument("--image", type=str,
                        help="The path to the input image", required=True)

    # parameter for Task 1 Localisation
    parser.add_argument("--task1-remove-extra-white", action="store_true",
                        help="Enable removing extra whitespace in task 1. By default it is disable.")
    parser.add_argument("--annotated-image-output-path", type=str,
                        help="If specified, output the image with annotation boxes in that path (overwrite existing). Please specify the full path including name")

    # parameter for Task 2 OCR
    parser.add_argument('--ocr-height', type=int, default=32,
                        help="input height for ocr model, default: 256 for resnet*, ""64 for inception")
    parser.add_argument('--ocr-width', type=int, default=196,
                        help="input width for ocr model, default: 128 for resnet*, ""256 for inception")
    parser.add_argument('--ocr_keep_ratio',action='store_true',
                        help='use if wanna keep image length fixed.')
    parser.add_argument('--ocr-voc-type', type=str, default='ALLCASES_SYMBOLS',
                       choices=['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'])    
    
    parser.add_argument('--ocr-output-path', type=str,
            help="Output the results of localisation & OCR to file in specified path (overwrite existing).")

    # parameter for Task 3 Key info extraction
    parser.add_argument('--extraction-output-path', type=str,
            help="Output the results of extraction to file in specified path (overwrite any existing file).")

    # hardware options
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

if __name__ == '__main__':
    args = get_args()

    # Guarding against bad arguments.
    
    if not osp.isfile(args.detection_model_path):
        raise ValueError("The path to the detection (task 1) model is wrong!")
    if not osp.isfile(args.ocr_model_path):
        raise ValueError("The path to the OCR (task 2) model is wrong!")
    if not osp.isfile(args.extraction_model_path):
        raise ValueError("The path to the extraction (task 3) model is wrong!")
    if not osp.isfile(args.image):
        raise ValueError("The path to the input image \""+args.image+"\" is wrong!")
    
    
    gpu_devices = list(range(torch.cuda.device_count()))
    if len(gpu_devices) != 0 and args.gpu_device not in gpu_devices:
        raise ValueError("Your GPU ID is out of the range! You may want to check with 'nvidia-smi'")
    elif args.use_cuda and not torch.cuda.is_available():
        raise ValueError("The argument --use-cuda is specified but it seems you cannot use CUDA!")
    
    task_1_model = task1.get_CTPN_model_default_config(args.detection_model_path)
    task_2_model = task2.getDefaultOCRModel(args.ocr_model_path, args.ocr_voc_type)
    task_3_model = task3.getDefaultModel(args.extraction_model_path)

    task_1_predict = task1.OneImagePrediction(task_1_model, args.task1_remove_extra_white,\
        args.use_cuda, args.use_amp, args.gpu_device)
    task_2_predict = task2.Prediction(task_2_model, args.ocr_voc_type)
    task_3_predict = task3.OneListPrediction(task_3_model, args.use_cuda, args.gpu_device)

    """
    image = Image.load(...)
    image = nparray(image)
    image = torch.Tensor(image).to(device)
    o1 = model1(image)
    texts = []
    for o in o1:
        cropped = image.crop(o1[i])
        # to tensor or something here
        o2 = model2(cropped)
        texts.append(o2)
    o3 = model3 (texts)
    pred[o3.class].append(o3)
    Output o1, o2, o3
    """

    image = Image.open(args.image).convert("RGB")
    image_np = np.array(image)
    detected_bboxes, bboxes_scores = task_1_predict.run(image_np)
    # We don't use bboxes_scores here
    detected_bboxes = detected_bboxes.tolist() # list of list of 4 value
    boxes = sorted(detected_bboxes)
    texts = []
    box_texts = []
    print("Localization done.")
    for box in boxes:
        box = [round(coord) for coord in box]
        cropped_image = image.crop(box)
        predicted_str = task_2_predict.predict_one_image(cropped_image,\
            args.ocr_width, args.ocr_height, args.ocr_keep_ratio)
        texts.append(predicted_str)

        # For output task 1+2
        box_texts.append((*box, predicted_str))
    print("OCR done.")
    result = task_3_predict.run(texts) # A dict
    print("Extraction done.")

    print(result) # To console

    if args.ocr_output_path is not None:
        with open(args.ocr_output_path, 'w') as f:
            for entry in box_texts:
                line = ','.join(map(str, entry))
                f.write(line + "\n")
    if args.extraction_output_path is not None:
        with open(args.extraction_output_path, 'w', encoding="utf-8") as f:
            json.dump(result, f, indent=4)

    image_with_annotation_box = draw_boxes(image = image, boxes = boxes)
    if args.annotated_image_output_path is not None:
        Image.fromarray(image_with_annotation_box).save(args.annotated_image_output_path)
    
    cv2.namedWindow("Output Image")
    cv2.imshow("Output Image", image_with_annotation_box)
    cv2.waitKey(0)
