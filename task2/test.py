import argparse
import torch
import torchvision.transforms as transforms
import numpy as np
import os
import csv
import sys

sys.path.append(os.getcwd())

from torchvision.transforms.functional import to_tensor
from task2.tools.utils import AlignCollate, get_vocabulary,data
from task2.model.srn_model import SRNModel
from PIL import Image

def get_args():
    parser = argparse.ArgumentParser(description="OCR softmax loss classification")
    parser.add_argument('--image-dir', type=str, metavar='PATH',
                        default="images", help="Path to images")
    parser.add_argument("--box-dir", type=str, metavar='PATH', default='boxes',
                        help="Bounding boxes txt path")
    parser.add_argument('--output-dir', type=str, metavar='PATH', default='output',
                        help="path for outputting prediction result")
    parser.add_argument('--height', type=int, default=64,
                        help="input height, default: 256 for resnet*, ""64 for inception")
    parser.add_argument('--width', type=int, default=256,
                        help="input width, default: 128 for resnet*, ""256 for inception")
    parser.add_argument('--trained-model',type=str,default='',help='the trained model')
    parser.add_argument('--keep_ratio',action='store_true',
                        help='use if wanna keep image length fixed.')
    parser.add_argument('--voc_type', type=str, default='LOWERCASE',
                       choices=['LOWERCASE', 'ALLCASES', 'ALLCASES_SYMBOLS'])
    # parser.add_argument('--voc_type', type=str, default='LOWERCASE')
    parser.add_argument('--cuda', type=str, default='0')
    parser.add_argument('--metric', type=str, default='acc')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max_len', type=int, default=25)

    #model config
    parser.add_argument('--in_channels',type=int,default=512,help='the srn_head input channel is same as the backbone output')
    parser.add_argument('--out_channels',type=int,default=38,help='the output logits dimension')
    # parser.add_argument('--max_len',type=int,default=25,help='the pvam decode steps')
    parser.add_argument('--num_heads',type=int,default=8,help='the Multihead attention head nums')
    parser.add_argument('--pvam_layer',type=int,default=2,help='the pvam default layers')
    parser.add_argument('--gsrm_layer',type=int,default=4,help='the gsrm default layers')
    parser.add_argument('--hidden_dims',type=int,default=512,help='d_model in transformer')
    return parser.parse_args()

def resizeAndNormalize (to_tensor, image, w, h, keep_ratio):
    imgH = h
    imgW = w
    if keep_ratio:
      ratio = w/float(h)
      imgW = int(np.floor(ratio * imgH))
      imgW = max(imgH * 1, imgW)  # assure imgH >= imgW
      imgW = min(imgW, 400)

    new_image = image.resize((imgW, imgH), Image.BILINEAR)
    new_image = to_tensor(new_image)
    new_image.sub_(0.5).div_(0.5)
    return new_image

def getDefaultOCRModel(trained_model_path, voc_type='LOWERCASE'):
    output_type ={'LOWERCASE':38,'ALLCASES':64,'ALLCASES_SYMBOLS':71}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ocr_model = SRNModel(512,output_type[voc_type],800,8,2,4,512)
    ocr_model.load_state_dict(torch.load(trained_model_path, map_location=device))
    return ocr_model

class Prediction():
    def __init__(self,model,voc_type='LOWERCASE'):
        # super(Prediction,self).__init__()
        self.model = model.eval()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.voc = get_vocabulary(voc_type)
        self.to_tensor = transforms.ToTensor()

    def decode(self,pred):
        pred_str = ''
        for i in range(len(pred)):
            if self.voc[pred[i]]=='EOS':
                break
            pred_str +=self.voc[pred[i]]
        while len(pred_str)>0 and pred_str.endswith('PADDING'):
            pred_str = pred_str[:-7]
        return pred_str

    def predict_one_image (self, image, w,h,keep_ratio) -> str:
        boxTensor = resizeAndNormalize(self.to_tensor, image, w, h, keep_ratio)
        image = boxTensor[None, :]
        image = image.to(self.device)
        output_dict, loss_dict = self.model(image, None)
        pred = output_dict['decoded_out'].view(-1).cpu().numpy()
        pred_str = self.decode(pred)
        return pred_str

    def run(self,image_dir,box_dir,output_dir,w,h,keep_ratio):
        self.model = self.model.eval()
        with torch.no_grad():
            possible_extension_image = ("jpg", "png", "jpeg", "JPG")

            files = os.listdir(image_dir)
            images = [f for f in files if f.endswith(possible_extension_image)]

            if len(images) == 0:
                raise ValueError("There are no images for prediction in ",image_path)

            # Đọc file từ image_path, lấy annotation từ box_path
            # Với từng box, crop image ra rồi predict từng cái

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
            # Sau đó output đống prediction trong 1 hình ra file txt
            # collate = AlignCollate(h, w, )

            for name in images:
                bare_name = os.path.splitext(name)[0]
                image_path = os.path.join(image_dir, name)
                anno_path = os.path.join(box_dir, bare_name+".txt")
                out_path = os.path.join(output_dir, bare_name+".txt")

                original_image = Image.open(image_path).convert("RGB")
                with open(anno_path, 'r') as fanno:
                    with open(out_path, 'w') as fout:
                        boxes = [] 
                        # We sort the box from left to right, then from top to bottom
                        # For the purpose of task 3: the order of the lines matter.
                        for line in csv.reader(fanno):
                            x_min, y_min, x_max, y_max = [int(c) for c in line[0:4]]
                            boxes.append(((x_min,y_min,x_max,y_max)))
                        boxes = sorted(boxes)
                        for box in boxes:
                            x_min, y_min, x_max, y_max = box
                            pred_str = self.predict_one_image(original_image.crop((x_min,y_min,x_max+1,y_max+1)),
                                w, h, keep_ratio)
                            fout.write(pred_str+"\n")

if __name__ =="__main__":
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_type ={'LOWERCASE':38,'ALLCASES':64,'ALLCASES_SYMBOLS':71}
    ocr_model = SRNModel(args.in_channels,output_type[args.voc_type],args.max_len,args.num_heads,args.pvam_layer,args.gsrm_layer,args.hidden_dims)
    ocr_model.load_state_dict(torch.load(args.trained_model, map_location=device))

    image_dir = args.image_dir
    if image_dir is None:
        raise "No image path provided!"
    box_dir = args.box_dir
    if box_dir is None:
        raise "No txt-box path provided!"
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    predictor = Prediction(ocr_model,args.voc_type)
    predictor.run(image_dir, box_dir, output_dir, args.width, args.height, args.keep_ratio)

