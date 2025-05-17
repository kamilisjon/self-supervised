# Installation (only tested with Ubuntu-22.04)

Pre-requisite: [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux)

```sh
git clone https://github.com/kamilisjon/self-supervised.git
cd self-supervised
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -e .
```

# Usage
1. Prepare data
  1. Download data from https://www.kaggle.com/datasets/volodymyrgavrysh/pneumothorax-binary-classification-task
  1. Extraxt the download zip file
  1. Generate pre-text data `python examples/data/pneumothorax.py`
  1. Vizualize examples `python examples/data/viz_classification_data.py`