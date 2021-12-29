import os
import sys
import json
import torch
import argparse
import os.path as osp
import torch.nn.functional as F

sys.path.append(os.getcwd())

from task3.data.datasets.sroie2019 import CustomDataset
from task3.data.datasets.sroie2019.variables import SROIE_TEXT_MAX_LENGTH, SROIE_VOCAB
from task3.utils.logger import create_logger
from task3.data.datasets.dataloader import create_dataloader

from zipfile import ZipFile, ZIP_DEFLATED
from task3.configs import configs
from task3.model.charlm import CharacterLevelCNNHighwayBiLSTM
from task3.data.postprocessing import convert_predictions_to_dict
from task3.data.datasets.sroie2019 import SROIE2019Dataset, TestBatchCollator
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description="Character-Level: Predictions")

    parser.add_argument("--config-file", action="store", help="The path to the yaml configs file.")
    parser.add_argument("--trained-model", action="store", help="The path to the trained model state dict file.")
    parser.add_argument("--use-cuda", action="store_true",
                        help="Enable or disable cuda during prediction. By default it is disable")
    parser.add_argument("--gpu-device", default=0, type=int, help="Specify the GPU id to use for the prediction.")
    parser.add_argument("--input-dir", action="store", help="The folder contains txt files for prediction.", required=True)
    parser.add_argument("--output-dir", action="store", help="The folder for storing the results.", required=True)
    return parser.parse_args()

def getDefaultModel(trained_model_path):
    model_params = dict(configs.MODEL.PARAMS)
    model = CharacterLevelCNNHighwayBiLSTM(n_classes=configs.DATASET.NUM_CLASSES,
            max_seq_length=SROIE_TEXT_MAX_LENGTH,
            char_vocab_size=len(SROIE_VOCAB),
            **model_params)
    checkpoint = torch.load(trained_model_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

def text2ArrayDefault(raw_text):
    """
    Convert a raw text to a one-hot character vectors.
    dataset.py for more customizable
    
    Args:
        raw_text: The input text.

    Returns:
        An array of one-hot character vectors whose shape is (text_max_length,)
        
    """
    if len(raw_text) == 0:
        raise ValueError("Cannot have an empty text!")

    data = []
    # Convert string into a vector of number, which is its order in the vocabulary.
    for i, char in enumerate(raw_text):
        letter2idx = SROIE_VOCAB.find(char) + 1  # +1 to avoid the confusion with the token padding value
        data.append(letter2idx)
    data = np.array(data, dtype=np.int64)

    # the length of the text array must be at most max_length
    # otherwise, pad them with zeros
    if len(data) > SROIE_TEXT_MAX_LENGTH:
        data = data[:SROIE_TEXT_MAX_LENGTH]
    elif 0 < len(data) < SROIE_TEXT_MAX_LENGTH:
        data = np.concatenate((data, np.zeros((SROIE_TEXT_MAX_LENGTH - len(data),), dtype=np.int64)))
    elif len(data) == 0:
        data = np.zeros((SROIE_TEXT_MAX_LENGTH,), dtype=np.int64)
    return data

class OneListPrediction:
    def __init__(self,model,use_cuda, gpu_device):

        # A boolean to check whether the user is able to use cuda or not.
        use_cuda = torch.cuda.is_available() and use_cuda
        
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

    # Raw text: list of str in a file (sorted in bounding box from left to right)
    # Returns: a dictionary where keys are "company", "address", "date", "total"
    @torch.no_grad()
    def run (self, raw_texts):
        self.model = self.model.eval()
        one_hot_text = []
        for line in raw_texts:
            one_hot_text.append(text2ArrayDefault(line.upper()))
        one_hot_text = torch.from_numpy(np.array(one_hot_text, dtype=np.int64))
        one_hot_texts = one_hot_text[None, :]
        one_hot_texts = one_hot_texts.to(self.device)
        outputs = self.model(one_hot_texts)

        batch_probs, batch_preds = torch.max(F.softmax(outputs, dim=2), dim=2)

        class_dict = dict(configs.DATASET.CLASS_NAMES)
        probs = batch_probs[0].tolist()
        preds = batch_preds[0].tolist()
        result = convert_predictions_to_dict(class_dict=class_dict,
                    raw_texts=raw_texts,
                    probabilities=probs,
                    predicted_classes=preds)

        return result

class PredictionDemo:
    def __init__(self,input_dir,trained_model_path, use_cuda, gpu_device):
        
        # A boolean to check whether the user is able to use cuda or not.
        use_cuda = torch.cuda.is_available() and use_cuda
        
        # The declaration and tensor type of the CPU/GPU device.
        if not use_cuda:
            self.device = torch.device("cpu")
            torch.set_default_tensor_type('torch.FloatTensor')
        else:
            self.device = torch.device("cuda")
            torch.cuda.set_device(gpu_device)
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            
            torch.backends.cudnn.enabled = True
        
        output_dir = os.path.normpath(configs.OUTPUT_DIR)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        
        self.logger = create_logger(name="Test phase", output_dir=output_dir, log_filename="test-log")
        
        self.logger.info("Creating the test dataset...")
        
        test_dataset = CustomDataset(input_dir)
        self.nb_images = len(test_dataset)
        
        # The declaration of the dataloader arguments.
        dataloader_args = dict(configs.DATALOADER.EVALUATION)
        dataloader_args["generator"] = torch.Generator(device=self.device)
        
        # Adding the collate_fn.
        dataloader_args["collate_fn"] = TestBatchCollator()
        self.test_loader = create_dataloader(dataset=test_dataset, is_train=False, **dataloader_args)
        self.logger.info("Initialisation and loading the model's weight...")
        
        self.trained_model = getDefaultModel(trained_model_path).to(self.device)
    
    @torch.no_grad()
    def run(self, output_dir):
        self.trained_model = self.trained_model.eval()
        
        list_raw_texts = []
        list_filenames = []
        
        probabilities = []
        predicted_classes = []
        
        dictStringToClassName = {0:"none", 1:"company", 2:"address", 3:"date", 4:"total"}
        
        with open(os.path.join(output_dir,"all_prediction.csv"), 'w') as fpred:
            fpred.write("line,predicted class,probability,filename\n")
            for i, batch_samples in enumerate(self.test_loader):
                one_hot_texts, raw_texts, filenames = batch_samples
                # print(raw_texts) # list of list of strings
                # print(filenames) # list of string, which is filename
                list_raw_texts.extend(raw_texts)
                list_filenames.extend(filenames)
                
                # text data shape: [N, L, C],
                # where N: the number of rows containing in one csv file, L:input length, C: vocab size
                one_hot_texts = one_hot_texts.to(self.device)
                
                outputs = self.trained_model(one_hot_texts)

                # print("outputs",outputs.size())
                
                batch_probs, batch_preds = torch.max(F.softmax(outputs, dim=2), dim=2)

                # print("batch_probs, batch_preds",batch_probs.size(),batch_preds.size())
                
                probabilities.extend(batch_probs.squeeze().tolist())
                predicted_classes.extend(batch_preds.squeeze().tolist())

                # print the line, predicted class, probability, filename into corresponding file
                for j in range(len(filenames)):
                    for k in range(len(raw_texts[j])):
                        fpred.write(raw_texts[j][k]+',') # 0 since always 1 file
                        fpred.write(dictStringToClassName[int(batch_preds[j,k].item())]+',')
                        fpred.write(str(batch_probs[j,k].item())+',')
                        fpred.write(filenames[j]+"\n")
        
        # Khúc này là xuất ra json
        class_dict = dict(configs.DATASET.CLASS_NAMES)
        
        for i, (probs, preds) in enumerate(zip(probabilities, predicted_classes)):
            raw_texts = list_raw_texts[i]
            filename = list_filenames[i]

            results = convert_predictions_to_dict(class_dict=class_dict,
                                                  raw_texts=raw_texts,
                                                  probabilities=probs,
                                                  predicted_classes=preds)
            
            self.logger.info("{0}/{1}: converting file {2} to json.".format(i + 1, self.nb_images, filename))
            
            with open(osp.join(output_dir, filename), "w", encoding="utf-8") as json_opened:
                json.dump(results, json_opened, indent=4)


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
    elif args.use_cuda and not torch.cuda.is_available():
        raise ValueError("The argument --use-cuda is specified but it seems you cannot use CUDA!")
    
    if args.config_file is not None:
        if not os.path.isfile(args.config_file):
            raise ValueError("The configs file is wrong!")
        else:
            configs.merge_from_file(args.config_file)
    configs.freeze()
    
    PredictionDemo(args.input_dir, args.trained_model, args.use_cuda, args.gpu_device).run(args.output_dir)
