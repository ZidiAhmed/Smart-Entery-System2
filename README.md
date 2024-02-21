Certainly! Below is a revised README.md file based on the information provided:

```markdown
# Smart Entry System

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![Stars](https://img.shields.io/github/stars/your-username/smart-entry-system.svg)](https://github.com/your-username/smart-entry-system/stargazers)
[![Forks](https://img.shields.io/github/forks/your-username/smart-entry-system.svg)](https://github.com/your-username/smart-entry-system/network/members)
[![Size](https://img.shields.io/github/repo-size/your-username/smart-entry-system)](https://github.com/your-username/smart-entry-system)
[![not Maintained](http://unmaintained.tech/badge.svg)](https://github.com/your-username/smart-entry-system)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Team/Participant Names:
- Zuwena Manor Almufadhly
- Rawuya Darwusg Albalusi
- Ahmed Said Alzeidi
- Fatima Khalfan Alhshmi
- Noof Nasser Alsiyabya

## Introduction

The Smart Entry System is a project developed by the above team/participants. It serves as an intelligent entry management system, utilizing computer vision and machine learning technologies implemented in Python. The system aims to enhance entry processes by automating facial recognition and mask detection.

## Installation

To run the Smart Entry System, follow the instructions below based on your operating system:

### Linux

Ensure you have Python3 installed on your machine. Install the required dependencies using the following commands in the terminal:
```bash
sudo pip3 install tensorflow
sudo pip3 install keras
sudo pip3 install sklearn
sudo pip3 install opencv-contrib-python
sudo pip3 install matplotlib
sudo pip3 install numpy
sudo pip3 install imutils
sudo pip3 install tk
```

### Windows

1. [Install Python](https://www.python.org/downloads/windows/), selecting the stable release.
2. During installation, check the option 'Add Python to environment variables.'
3. Disable 'Python' in 'Manage App Execution Aliases.'
4. Install dependencies using the following commands in the Command Prompt (CMD):
   ```bash
   python -m pip install tensorflow
   python -m pip install keras
   python -m pip install sklearn
   python -m pip install opencv-contrib-python
   python -m pip install matplotlib
   python -m pip install numpy
   python -m pip install imutils
   python -m pip install tk
   ```

### macOS

Install a newer version of Python from [here](https://www.python.org/downloads/macos/). Then, use the following command in the terminal to install dependencies:
```bash
pip install tensorflow
pip install keras
pip install sklearn
pip install opencv-contrib-python
pip install matplotlib
pip install numpy
pip install imutils
pip install tk
```

## How to Use

### GUI

The primary way to interact with the Smart Entry System is through the graphical user interface (`gui.py`). Run the GUI script to perform various tasks, including training the model, detecting masks in static images, and real-time video processing.

```bash
python3 gui.py
```

### Training the Model

1. Navigate to the 'Settings' tab in the GUI.
2. Click 'Browse' to select the dataset path.
3. Go to the 'Training' tab and click 'Train model' to initiate the training process.

Alternatively, use the following commands based on your OS:

- Linux:
  ```bash
  python3 train_mask_detector.py --dataset dataset
  ```
- Windows and macOS:
  ```bash
  python train_mask_detector.py --dataset dataset
  ```

### Face Mask Detection for Static Images

Use the GUI under the 'Image' tab or run the following commands:

- Linux:
  ```bash
  python3 detect_mask_image.py --image examples/ex1
  ```
- Windows and macOS:
  ```bash
  python detect_mask_image.py --image examples/ex1
  ```

### Face Mask Detection for Real-time Video

Utilize the GUI in the 'Video' tab or run the following commands:

- Linux:
  ```bash
  python3 detect_mask_video.py
  ```
- Windows and macOS:
  ```bash
  python detect_mask_video.py
  ```

## Acknowledgments

We would like to acknowledge the following contributors for their valuable contributions to the Smart Entry System:

- [Zuwena Manor Almufadhly](https://github.com/ZuwenaManorAlmufadhly)
- [Rawuya Darwusg Albalusi](https://github.com/RawuyaDarwusgAlbalusi)
- [Ahmed Said Alzeidi](https://github.com/AhmedSaidAlzeidi)
- [Fatima Khalfan Alhshmi](https://github.com/FatimaKhalfanAlhshmi)
- [Noof Nasser Alsiyabya](https://github.com/NoofNasserAlsiyabya)

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).

```

Please replace "your-username" in the badge URLs with the appropriate GitHub username once you have the project hosted on GitHub.# Smart-Entery-System2
