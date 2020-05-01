# Handwritten Japanese Deep Learning based OCR  with Touch panel demo
This is a handwritten Japanese OCR demo program based on a sample program from [**Intel(r) Distribution of OpenVINO(tm) Toolkit 2020.2**](https://software.intel.com/en-us/openvino-toolkit) (`handwritten-japanese-recognition.py`)  
The demo program has simple UI and you can write Japanese on the screen with touch panel by your finger tip and try Japanese OCR performance.  
The demo uses a pre-trained text-detection DL model from Intel(r) [Open Model Zoo](https://github.com/opencv/open_model_zoo) to detect the text region from the canvas and run DL based OCR for those texts.  
手書き日本語OCRデモです。[**Intel(r) Distribution of OpenVINO(tm) toolkit 2020.2**](https://software.intel.com/en-us/openvino-toolkit)に付属の`handwritten-japanese-recognition.py`デモを大幅に書き換えています。  
簡単なUIを用意していますのでタッチパネル付きPCがあれば指で字を書いて認識させるデモを行うことが可能です。  
Intel(r) [Open Model Zoo](https://github.com/opencv/open_model_zoo)の文字検出DLモデルで自動領域識別も行ない、DL-OCRモデルで文字認識を行います。  

![OCR demo](./resources/ocr-demo.gif)  

### Required DL models to run this demo

The demo expects the following model in the Intermediate Representation (IR) format:

   * handwritten-japanese-recognition-0001
   * text-detection-0003

You can download those models from OpenVINO [Open Model Zoo](https://github.com/opencv/open_model_zoo).
In the `models.lst` are the list of appropriate models for this demo
that can be obtained via `Model downloader`.
Please see more information about `Model downloader` [here](../../../tools/downloader/README.md).

## How to run

### 1. Install dependencies  
The demo depends on:
- opencv-python
- numpy

To install all the required Python modules you can use:

``` sh
(Linux) pip3 install -r requirements.txt
(Win10) pip install -r requirements.txt
```

### 2. Download DL models from OMZ
Use `Model Downloader` to download required models.
``` sh
(Linux) python3 $INTEL_OPENVINO_DIR/deployment_tools/tools/model_downloader/downloader.py --list models.lst
(Win10) python "%INTEL_OPENVINO_DIR%\deployment_tools\tools\model_downloader\downloader.py" --list models.lst
```

### 3. Run the demo app
This program doesn't accept any command line arguments. All file names and paths are hard coded in the source code.
``` sh
(Linux) python3 handwritten-japanese-OCR-touch-panel-demo.py
(Win10) python handwritten-japanese-OCR-touch-panel-demo.py
```

Please make sure following files are placed at proper location.
```
./  
+ handwritten_japanese-OCR-touch-panel-demo.py  
+ data  
| + kondate_nakayosi_char_list.txt  
+ intel  
| + handwritten-japanese-recognition-0001  
| | + FP16  
| | | + handwritten-japanese-recognition-0001.xml  
| | | + handwritten-japanese-recognition-0001.bin  
| + text-detection-0003  
| | + FP16  
| | | + text-detection-0003.xml  
| | | + text-detection-0003.bin  
```

## Demo Output
The application uses the terminal to show resulting recognition text and inference performance.

## Tested environment
- Windows 10 x64 1909
- Intel(r) Distribution of OpenVINO(tm) toolkit 2020.2
- Python 3.6.5 x64

## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
