# GNN for Fraud detection

## 1. Input data format:

~~~
- user : [str] = None
- card : [str] = None
- year : [str] = None
- month : [str] = None
- day : [str] = None
- time : [str] = None
- amount: [str] = None
- use_chip: Optional[str] = None
- merchant_name: Optional[str] = None
- merchant_city: Optional[str] = None
- merchant_state: Optional[str] = None
- zip : Optional[str] = None
- mcc : str = None
- errors : Optional[str] = None
~~~

## 2. Create Conda environment and Install necessary libraries

~~~
conda create -n fraud_detection python=3.11
conda activate fraud_detection
pip install requirements.txt
~~~

## 3. Inference
~~~
python infer.py
~~~
