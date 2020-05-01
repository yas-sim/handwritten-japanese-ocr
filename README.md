# Handwritten Japanese OCR  with Touch panel demo
This is a handwritten Japanese OCR demo program based on a sample program from OpenVINO 2020.2 (handwritten-japanese-recognition.py)  
The demo program has simple UI and you can write Japanese on the screen with touch panel by your finger tip and try Japanese OCR performance.
The demo uses pre-trained text-detection from Intel Open Model Zoo (OMZ) to detect text region from the canvas and run OCR for those texts.

## Required DL models to run this demo

The demo expects the following model in the Intermediate Representation (IR) format:

   * handwritten-japanese-recognition-0001
   * text-detection-0003

You can download those models from OpenVINO Open Model Zoo.
In the `models.lst` are the list of appropriate models for this demo
that can be obtained via `Model downloader`.
Please see more information about `Model downloader` [here](../../../tools/downloader/README.md).

### How to run

#### 1. Install dependencies  
The demo depends on:
- opencv-python
- numpy

To install all the required Python modules you can use:

``` sh
pip install -r requirements.txt
```

#### 2. Download DL models from OMZ
Use `Model Downloader` to download required models.
``` sh
python3 $INTEL_OPENVINO_DIR/deployment_tools/tools/model_downloader/downloader.py --list models.lst
```

#### 3. Run the demo app
This program doesn't accept any command line arguments. All file names and paths are hard coded in the source code.
``` sh
python3 handwritten-japanese-OCR-touch-panel-demo.py
```

Please make sure following files are placed at proper location.
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

## Demo Output
The application uses the terminal to show resulting recognition text and inference performance.


## See Also
* [Using Open Model Zoo demos](../../README.md)
* [Model Optimizer](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_Deep_Learning_Model_Optimizer_DevGuide.html)
* [Model Downloader](../../../tools/downloader/README.md)
