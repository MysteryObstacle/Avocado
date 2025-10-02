# Avocado🥑
An Interpretable Fine-Grained Intrusion Detection Model for Advanced Industrial Control Network Attacks

![](./img/Avocado.svg)

## 1. Introduction
XXX.

## 2. Preparation
### 2.1 Clone the repository
```bash
git clone https://github.com/MysteryObstacle/Avocado.git
cd Avocado
```
### 2.2 Create a new conda environment
```bash
conda create -n Avocado python=3.10.14
conda activate Avocado
```
### 2.3 Install Pytorch
```bash
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
```
### 2.4 Install other dependencies
```bash
pip install -r requirements.txt
```
### 2.5 Configure environment variables
If you are in Linux, you can do this:
```bash
export PYTHON PATH="${PYTHONPATH}:./"
```
If you are in Windows, you can add the Avocado root path to the python environment variable by hand.

## 3. Dataset
### 3.1 Original Dataset
| Dataset       | Packet | Label |
|---------------|--------|-------|
| CLA-M221      | ?      | ?     |
| GAS           | ?      | ?     |
| GRFICS        | ?      | ?     |
1. [CLA-M221]()
2. [GAS]()
3. [GRFICS]()
4. Or you can also prepare your own datasets.
### 3.2 Processed Dataset
After processing the data into the datasets folder, the file tree example is as follows:
```
├─CLA-M221
│  └─pcap
│     └─dataExecution+noisePadding
│     └─traditional_attack
│     └─training
│  └─deal
│     └─dataExecution+noisePadding
│     └─traditional_attack
│     └─training
│  └─data
│     └─train.parquet
│     └─test.parquet
└─GAS
   └─csv
      └─IanRawDataset.csv
   └─deal
      └─IanRawDataset.json.gz
      └─IanRawDataset_SUCCESS      
   └─data
      └─train.parquet
      └─test.parquet
```

## 3. Usage
### 3.1 Preprocess
```bash
python preprocess.py -t pcap -r datasets/CLA-M221
python preprocess.py -t csv -r datasets/GAS
```
### 3.2 Generate Dataset
```bash
python generate_dataset.py -r datasets/CLA-M221
python generate_dataset.py -r datasets/GAS
```
### 3.3 Train
```bash
python train.py -t train_avocado_clam221 -d datasets/CLA-M221/data -o outputs/avocado -l logs/avocado -c configs/avocado.json
python train.py -m avocado_q -t train_avocado_clam221 -d datasets/CLA-M221/data -o outputs/avocado_q -l logs/avocado_q -c configs/avocado_q.json

python train.py -m avocado_q -t train_avocado_gas -d datasets/GAS/data -o outputs/avocado_q_gas -l logs/avocado_q_gas -c configs/avocado_q_gas.json
```
### 3.3 Evaluate
```bash
python evaluate.py \ 
  -f datasets/CLA-M221/pcap/dataExecution+noisePadding/testPLC_attack+unknown.pcapng \
  -w outputs/avocado/model.pth \
  -o outputs/avocado/ \
  -d 2

python evaluate.py \
  -m avocado_q \
  -f datasets/CLA-M221/pcap/dataExecution+noisePadding/testPLC_attack+unknown.pcapng \
  -w outputs/avocado_q/model.pth \
  -o outputs/avocado_q/ \
  -d -1

python evaluate.py \
  -m avocado_q \
  -t parquet -f \
  datasets/GAS/data/test.parquet \
  -w outputs/avocado_q_gas/model.pth \
  -n 8 \
  -o outputs/avocado_q_gas/ \
  -d -1
```
