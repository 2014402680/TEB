#  Ensemble Graph Neural Network for TCR-Epitope Binding Prediction

Welcome to Ensemble Graph Neural Network for TCR-Epitope Binding Prediction.


## Folder Structure

The project's folder structure is as follows:

- **processed_data** folder:

  This folder contains the raw data for each dataset and the pre-processed 5-fold data. These data are used for training and testing the models.

- **results** folder:

  In this folder, we store the model's predictions on the datasets. These results can help us analyze model performance and generate further visualizations and reports.

## Quick Start

1. **Create a Conda Environment**:

   Start by creating a Conda environment with Python 3.11. If you haven't already installed Conda, you can get it from [Anaconda](https://www.anaconda.com/products/individual).

   ```shell
   conda create -n TEB python=3.11
   ```

   Activate the environment:

   ```shell
   conda activate TEB
   ```

2. **Install Dependencies**:

   Use pip to install the required packages listed in the requirements.txt file.

   ```shell
   pip install -r requirements.txt
   ```



 **Additional Information**:

   For more details and customization options, please refer to ours paper.
Have fun exploring the TEB framework!



## How to Train


Next, simply run the following command:

```bash
python train.py --gpu 0 --configs_path configs/pMTnet.yml --droup_out 0.1 --split StrictTCR
```

Please ensure that the paths in `configs/XXXXX.yml` are correct, including the paths for training and testing files, and the embeddings.
