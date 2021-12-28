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
    parser.add_argument("--gt-dir", type=str, metavar='PATH', default='gt',
                        help="Bounding boxes & ground truth txt directory")
    parser.add_argument('--correct-output-dir', type=str, metavar='PATH', default='output-correct',
                        help="folder for outputting correct prediction result")
    parser.add_argument('--incorrect-output-dir', type=str, metavar='PATH', default='output-incorrect',
                        help="folder for outputting incorrect prediction result")

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

class CustomEvaluation():
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

    @torch.no_grad()
    def run(self,image_dir,gt_dir,correct_output_dir,incorrect_output_dir,w,h,keep_ratio):
        self.model = self.model.eval()

        possible_extension_image = ("jpg", "png", "jpeg", "JPG")

        files = os.listdir(image_dir)
        images = [f for f in files if f.endswith(possible_extension_image)]

        if len(images) == 0:
            raise ValueError("There are no images for prediction in ",image_dir)

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

        n_correct = 0
        n_incorrect = 0

        for name in images:
            print('Entry: ',name)
            bare_name = os.path.splitext(name)[0]
            ext = os.path.splitext(name)[-1]
            image_path = os.path.join(image_dir, name)
            anno_path = os.path.join(gt_dir, bare_name+".txt")
            correct_out_txt_path = os.path.join(correct_output_dir, bare_name)
            incorrect_out_txt_path = os.path.join(incorrect_output_dir, bare_name)
            correct_out_img_path = os.path.join(correct_output_dir, bare_name)
            incorrect_out_img_path = os.path.join(incorrect_output_dir, bare_name)

            original_image = Image.open(image_path).convert("RGB")
            n_count = 0
            with open(anno_path, 'r', encoding='unicode_escape') as fanno:
                # We sort the box from left to right, then from top to bottom
                # For the purpose of task 3: the order of the lines matter.
                n_count += 1
                for line in fanno:
                    line=line.strip()
                    if len(line) == 0:
                        continue
                    line = line.split(",", 8)
                    x1,y1,x2,y2,x3,y3,x4,y4 = [int(c) for c in line[0:8]]
                    gt_str = line[-1]
                    x_min = min(x1,x2,x3,x4)
                    x_max = max(x1,x2,x3,x4)
                    y_min = min(y1,y2,y3,y4)
                    y_max = max(y1,y2,y3,y4)
                    cropped_image = original_image.crop((x_min,y_min,x_max+1,y_max+1))
                    pred_str = self.predict_one_image(cropped_image, w, h, keep_ratio)
                    if gt_str.strip() == pred_str.strip():
                        cropped_image.save(correct_out_img_path+"-"+str(n_count)+ext)
                        with open(correct_out_txt_path+"-"+str(n_count)+".txt", 'w') as f:
                            f.write("Gt: "+gt_str+"\n")
                            f.write("pred: "+pred_str+"\n")
                        n_correct += 1
                    else:
                        cropped_image.save(incorrect_out_img_path+"-"+str(n_count)+ext)
                        with open(incorrect_out_txt_path+"-"+str(n_count)+".txt", 'w') as f:
                            f.write("Gt: "+gt_str+"\n")
                            f.write("pred: "+pred_str+"\n")
                        n_incorrect += 1
        print('#correct #incorrect #acc',n_correct,n_incorrect,n_correct / (n_correct+n_incorrect))  

if __name__ =="__main__":
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_type ={'LOWERCASE':38,'ALLCASES':64,'ALLCASES_SYMBOLS':71}
    ocr_model = SRNModel(args.in_channels,output_type[args.voc_type],args.max_len,args.num_heads,args.pvam_layer,args.gsrm_layer,args.hidden_dims)
    ocr_model.load_state_dict(torch.load(args.trained_model, map_location=device))

    image_dir = args.image_dir
    if image_dir is None:
        raise "No image path provided!"
    gt_dir = args.gt_dir
    if gt_dir is None:
        raise "No ground truth path provided!"

    correct_output_dir = args.correct_output_dir
    if not os.path.exists(correct_output_dir):
        os.makedirs(correct_output_dir)
    incorrect_output_dir = args.incorrect_output_dir
    if not os.path.exists(incorrect_output_dir):
        os.makedirs(incorrect_output_dir)

    predictor = CustomEvaluation(ocr_model,args.voc_type)
    predictor.run(image_dir, gt_dir, correct_output_dir, incorrect_output_dir, args.width, args.height, args.keep_ratio)

