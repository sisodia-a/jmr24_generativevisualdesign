{\rtf1\ansi\ansicpg1252\cocoartf2709
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;\f1\fnil\fcharset0 .AppleSystemUIFontMonospaced-Regular;}
{\colortbl;\red255\green255\blue255;\red0\green0\blue0;\red24\green26\blue30;\red255\green255\blue255;
}
{\*\expandedcolortbl;;\cssrgb\c0\c0\c0;\cssrgb\c12157\c13725\c15686;\cssrgb\c100000\c100000\c100000;
}
\margl1440\margr1440\vieww24560\viewh12780\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs28 \cf0 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 ## Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis\
Code to replicate results in, "\outl0\strokewidth0 Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis\outl0\strokewidth0 \strokec2 ."  \
\
If you find this work useful, please use the following citation information in plaintext or BibTeX format:\
\
Sisodia, A, Burnap, A, and Kumar, V (2024). \outl0\strokewidth0 Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis\outl0\strokewidth0 \strokec2 . Journal of Marketing Research (under review).\
\
```\
@article\{sisodia2024disentangelment,\
  title= \outl0\strokewidth0 Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis\outl0\strokewidth0 \strokec2 \},\
  author=\{\outl0\strokewidth0 Sisodia, A, Burnap, A, and Kumar, V\outl0\strokewidth0 \strokec2 \},\
  journal=\{\outl0\strokewidth0 Journal of Marketing Research\outl0\strokewidth0 \strokec2  (under review)\},\
  volume=\{XXX\},\
  number=\{XXX\},\
  pages=\{XXX\},\
  year=\{2024\},\
  publisher=\{XXX\}\
\}\
```\
\
## Usage\
\
## Installation\
#### Libraries\
This codebase was built using the Pytorch neural net framework.  However, it also depends on over 100 other Python packages, as well as GPU libraries including CUDA runtimes and cuDNN kernel support. We can not guarantee this code will run and replicate unless the same environment and library versions are used.  To install these packages (including the same library versions), please use the following:\
\
    conda env create -f disentanglement_env.yml\
\
#### Data\
To download the dataset, please use the following links. Copy these files to \'91./hyperparameter_selection/data/watches/\'91 and \'91./post_model_search/data/watches/\'91 directories.\
\
* #### [Watches Dataset](https://www.dropbox.com/scl/fo/akj3w8pat0lg1fa4ax480/h?rlkey=5d4ykq5br3kzkwarhi4ld4na8&dl=0)\
\
\pard\pardeftab720\partightenfactor0
\cf0 \outl0\strokewidth0 ##
\fs26 #
\fs28  Run\
\pard\pardeftab720\partightenfactor0
\cf0 \outl0\strokewidth0 \strokec2 \
\pard\pardeftab720\partightenfactor0
\cf0 Use `python main.py <\outl0\strokewidth0 param\outl0\strokewidth0 \strokec2 -name> <param-value>`. For example:\
\pard\pardeftab720\partightenfactor0
\cf0 \
\pard\pardeftab720\partightenfactor0
\cf0 \outl0\strokewidth0 ```\
python main.py --sup_signal continuousprice -s 1 --name continuousprice_s1 --btcvae-B 1 --btcvae-M 16\
```\outl0\strokewidth0 \strokec2 \
\pard\pardeftab720\partightenfactor0
\cf0 \
\pard\pardeftab720\partightenfactor0

\fs26 \cf0 ### Output\
\pard\pardeftab720\partightenfactor0

\fs28 \cf0 \
\pard\pardeftab720\partightenfactor0

\fs26 \cf0 This will create a directory `results/<model-name>/` which will contain:\
\
* **specs.json**: The parameters used to run the program (default and modified with CLI).\
* **train_losses.csv**: All (sub-)losses computed during training on the train and validation dataset.\
* **test_losses.log**: All (sub-)losses computed at the end of training on the test1 and test2 dataset. \
\pard\pardeftab720\partightenfactor0
\cf0 \outl0\strokewidth0 * **reconstruct_traverse.png**: latent traversals of latent dimensions. \
\pard\pardeftab720\partightenfactor0
\cf0 * **filename_test1.csv**: filenames of all watches in the test1 dataset. \
* **filename_test2.csv**: filenames of all watches in the test2 dataset. \
\pard\pardeftab720\partightenfactor0
\cf0 * **filename_train.csv**: filenames of all watches in the train dataset. \
* **mean_params_test1.csv**: mean visual characteristics of all watches in the test1 dataset. \
* **mean_params_test2.csv**: mean visual characteristics of all watches in the test2 dataset. \
* **mean_params_train.csv**: mean visual characteristics of all watches in the train dataset. \
\pard\pardeftab720\partightenfactor0

\fs28 \cf0 \outl0\strokewidth0 \strokec2 \
\pard\pardeftab720\partightenfactor0

\fs26 \cf0 ### Help
\fs28 \outl0\strokewidth0 \

\fs26 \outl0\strokewidth0 \strokec2 \
```\
usage: main.py ...\
\
General options:\
  name						Name of the model for storing or loading purposes.\
  -s, --seed SEED				Random seed.\
  -e, --epochs EPOCHS			Maximum number of epochs to run for.\
  -b, --batch-size BATCH_SIZE	Batch size for training.\
  --lr LR						Learning rate.\
  -z, --latent-dim LATENT_DIM	Dimension of the latent variable.\
\pard\pardeftab720\partightenfactor0
\cf0 \outl0\strokewidth0   \'97tv --threshold-val			Threshold for Masking.\
\pard\pardeftab720\partightenfactor0
\cf0   --sup_signal				Choice of Signal. (Examples: discreteprice, brand, circa, material, movement or a combination in the format <signal1>_<signal2> etc.)\
\pard\pardeftab720\partightenfactor0
\cf0 \outl0\strokewidth0 \strokec2 \
Loss specific hyperparameters:\
  --btcvae-A BTCVAE_A			Weight of the MI term (alpha in the paper).\
  --btcvae-G BTCVAE_G			Weight of the dim-wise KL term (gamma in the paper).\
  --btcvae-B BTCVAE_B			Weight of the TC term (beta in the paper).\
\pard\pardeftab720\partightenfactor0
\cf0 \outl0\strokewidth0   --btcvae-M BTCVAE_M			Weight of the supervised loss term (delta in the paper).\
\pard\pardeftab720\partightenfactor0
\cf0 \outl0\strokewidth0 \strokec2 \
```\
\pard\pardeftab720\partightenfactor0

\fs28 \cf0 \
\pard\pardeftab720\partightenfactor0

\fs26 \cf0 \outl0\strokewidth0 ### Hyperparameter Selection\
\
Run the same model configuration (i.e. a combination of the supervisory signal and loss-specific hyperparameters) for different random seeds. Select the optimal loss-specific hyperparameters for a particular supervisory signal combination based on the lowest supervised loss on the validation dataset (averaged across different seeds).\
\
\pard\pardeftab720\partightenfactor0

\fs28 \cf0 \
\pard\pardeftab720\partightenfactor0

\fs26 \cf0 ### UDR Calculation\
\pard\pardeftab720\partightenfactor0

\fs28 \cf0 \outl0\strokewidth0 \strokec2 \
\pard\pardeftab720\partightenfactor0

\fs26 \cf0 \outl0\strokewidth0 Run the most optimal loss-specific hyperparameters for each combination of the supervisory signal across different random seeds. Copy the files stored in \'91results/<model-name>/\'91 directory with the filename ending in \'91mean_params_test2.csv\'92 to the \'91calculate_udr\'92 folder. Switch to an R environment and calculate UDR using the below command. You may need to modify the R script and adjust the model name.\
\
\pard\pardeftab720\partightenfactor0

\fs28 \cf0 \
```\
\pard\pardeftab720\partightenfactor0

\fs26 \cf0 Rscript udr_calculation.R --sup_signal='brand'
\fs28 \
```\
\pard\pardeftab720\partightenfactor0
\cf0 \outl0\strokewidth0 \strokec2 \
\pard\pardeftab720\partightenfactor0
\cf0 ## Citation\
Feel free to use this code for your research. If you find this code useful, please use the following citation information in plaintext or BibTeX format:\
\
\pard\pardeftab720\partightenfactor0
\cf0 \outl0\strokewidth0 Sisodia, A, Burnap, A, and Kumar, V (2024). Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis. Journal of Marketing Research (under review).\
\pard\pardeftab720\partightenfactor0
\cf0 \outl0\strokewidth0 \strokec2 \
```\
\pard\pardeftab720\partightenfactor0
\cf0 \outl0\strokewidth0 @article\{sisodia2024disentangelment,\
  title= Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis\},\
  author=\{Sisodia, A, Burnap, A, and Kumar, V\},\
  journal=\{Journal of Marketing Research (under review)\},\
  volume=\{XXX\},\
  number=\{XXX\},\
  pages=\{XXX\},\
  year=\{2024\},\
  publisher=\{XXX\}\
\}\
\pard\pardeftab720\partightenfactor0
\cf0 \outl0\strokewidth0 \strokec2 ```\
\
## Acknowledgments\
Portions of this codebase were built on elements from the following open source projects, please see their respective licenses and cite accordingly:\
* [disentangling-vae](https://github.com/YannDubs/disentangling-vae)\
\
\pard\pardeftab720\partightenfactor0
\cf0 \outl0\strokewidth0 ```\
\pard\pardeftab720\partightenfactor0

\f1 \cf3 \cb4 \outl0\strokewidth0 \strokec3 @misc\{dubois2019dvae,\
  title        = \{Disentangling VAE\},\
  author       = \{Dubois, Yann and Kastanos, Alexandros and Lines, Dave and Melman, Bart\},\
  month        = \{march\},\
  year         = \{2019\},\
  howpublished = \{\\url\{http://github.com/YannDubs/disentangling-vae/\}\}\
\}
\f0 \cb1 \
\pard\pardeftab720\partightenfactor0
\cf0 \outl0\strokewidth0 ```\
}
