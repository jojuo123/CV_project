# Receipt reader: Documentation

## Prologue

We can run the code at https://colab.research.google.com/drive/1X3BE_n7FzlpilzUP0uEQ6JlR2LRzVwOT

The source code can be found at https://github.com/jojuo123/CV_project/tree/v1

Before batch evaluation or single prediction, we should download the models.

Model task 1 (BlackStar1313's pre-trained model): https://github.com/BlackStar1313/ICDAR-2019-RRC-SROIE/releases/download/v1.0/CTPN_FINAL_CHECKPOINT.pth

Our trained task 2 model: https://github.com/jojuo123/CV_project/releases/download/v1/task2.pth

Our trained task 3 model: https://github.com/jojuo123/CV_project/releases/download/v1/task3.pth

All corresponding executable python files are provided with `-h` command line to print out the help for the arguments.

## Requirement

Python3

## Task 1 (Text detection) batch evaluation

```sh
python task1/demo.py --trained-model <path/to/task1model/CTPN_CHECKPOINT.pth> \
--image-folder <input_folder> --output-folder <output_folder>
```

The program will read every image files inside `input_folder`, for each input file, the program produces two files with the same name and but different extensions into `output_folder`: an image which is original image with annotation box, and a text file where each line is a comma-separated quintuple describe a detected text box with the following format:

```
x_min,y_min,x_max,y_max, prob
x_min,y_min,x_max,y_max, prob
x_min,y_min,x_max,y_max, prob
...
```

For more parameters, user may invoke `python task1/demo.py -h` for help.

## Task 2 (OCR) batch evaluation

```sh
python task2/test.py \
  --image-dir <image_dir> \
  --box-dir <box_dir> \
  --output-dir <output_dir> \
  --trained-model <path/to/task2.pth> \
  --height 32\
  --width 200\
  --voc_type ALLCASES_SYMBOLS \
  --max_len 800 \
```

For each image `X.[jpg|png|jpeg]` in `image_dir`, the program looks for `X.txt`, it assumes that each line in `X.txt` contains comma-separated detected boxes in the corresponding image (same format as the output of task 1). For each portion of the image, the program predicts a string. All predicted strings are outputted line-by-line to `X.txt` in the `output_dir`. Since the names are the same as those in `box_dir`, `box_dir`, and `output_dir` must be different.

The `height`, `width`, `voc_type`, and `max_len` parameters are fine-tuned for SROIE 2019 dataset, we recommend using the same parameter for compatibility with our trained model.

There is the `keep_ratio` option. Users may invoke `python task2/test.py -h` for more detail.

## Task 3 (Information extraction) batch evaluation

```sh
python task3/demo.py --trained-model <path/to/task3.pth> \
--input-dir <input_dir> --output-dir <output_dir>
```

For each text file `X.txt` in `input_dir`, the program gives a prediction `X.txt` in `output_dir` which is in JSON. The program assumes that each line in the input text file is a detected text line, in other words, in the same format as output files of task 2 batch evaluation above. The line must contain a maximum of 68 characters (limit of SROIE 2019 dataset).

## Read your receipt image

```sh
python main.py --detection-model-path <path/to/task1/model.pth> \
--ocr-model-path <path/to/task2.pth> \
--extraction-model-path <path/to/task3.pth> \
--image <path/to/your/image.jpg> \
--ocr-output-path <path/to/ocr/output.txt> \
--extraction-output-path <path/to/extraction/output.txt> \
--annotated-image-output-path <path/to/annotated/image/output.json>
```

`detection-model-path`, `ocr-model-path`, `extraction-model-path` is self-descriptive. Provide the path to your image in the `image` parameter. Parameters `ocr-output-path`, `extraction-output-path`, `annotated-image-output-path` are all optional for outputting intermediate steps. After execution, the annotated image is shown and the JSON-formatted extracted information is printed in the command line.

For more optional parameters, invoke `python main.py -h` for details.
