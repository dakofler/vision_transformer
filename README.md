# Vision Transformer

An Implemenentation of the paper [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929) in Pytorch.

## Installation

- Clone the repository
```bash
git clone https://github.com/dakofler/vision_transformer.git
cd vision_transformer/
```
- (Optional) Create a Python virtual environment (Linux)
```bash
python3 -m venv .venv
source .venv/bin/activate
```
- Install dependencies
```bash
pip install -r requirements.txt
```

## Usage
- Download the [CIFAR10-10](https://www.cs.toronto.edu/~kriz/cifar.html)
- Upack and put the files into a `./data` directory

```bash
vision_transformer/
├─ data/
│  ├─ batches.meta
│  ├─ data_batch_1
│  ├─ data_batch_2
│  ├─ data_batch_3
│  ├─ data_batch_4
│  ├─ data_batch_5
│  ├─ readme.html
│  ├─ test_batch
...
```
- run the training script

```bash
python3 train.py
```

## Author
Daniel Kofler - [dkofler@outlook.com](mailto:dkofler@outlook.com)<br>
2025