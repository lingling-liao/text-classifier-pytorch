# Text Classifier PyTorch
Text classification using PyTorch.

## Official Docker images for TensorFlow

Docker pull command:

```
docker pull tensorflow/tensorflow:2.3.0-gpu-jupyter
```

Running containers:

```
docker run --gpus all -p 6006:6006 -p 8888:8888 -v [local]:/tf -itd tensorflow/tensorflow:2.3.0-gpu-jupyter
```

## Using PyTorch

Install torch and torchvision

```
pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 -f https://download.pytorch.org/whl/torch_stable.html
```

Install transformers and others

```
pip install openpyxl==3.0.5 pandas==1.1.5 scikit-learn==0.24.0 transformers==4.1.1 xlrd==2.0.1
```

## Training a model

Running the command:

```
python train_text_classifier.py
```
