<p align="center"><img src="./conjoint_analysis_and_ideal_point_design/data_conjointanalysis/disentanglement_example.png" alt="disentanglement_example" style="display: block; margin-left: auto; margin-right: auto; width: 100%; text-align: center;" > </p>

## Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis
Code to replicate results in, "Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis."  

If you find this work useful, please use the following citation information in plaintext or BibTeX format:

Sisodia, A, Burnap, A, and Kumar, V (2024). Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis. Journal of Marketing Research (under review).

```
@article{sisodia2024disentangelment,
  title= Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis},
  author={Sisodia, A, Burnap, A, and Kumar, V},
  journal={Journal of Marketing Research (under review)},
  volume={XXX},
  number={XXX},
  pages={XXX},
  year={2024},
  publisher={XXX}
}
```

## Installation
#### Libraries
This codebase was built using the Pytorch neural net framework.  However, it also depends on over 100 other Python packages, as well as GPU libraries including CUDA runtimes and cuDNN kernel support. We can not guarantee this code will run and replicate unless the same environment and library versions are used.  To install these packages (including the same library versions), please use the following:

    conda env create -f disentanglement_env.yml

#### Data
To download the dataset, please use the following links. Copy these files to ‘./hyperparameter_selection/data/watches/‘ and ‘./post_model_search/data/watches/‘ directories.

* #### [Watches Dataset](https://www.dropbox.com/scl/fo/akj3w8pat0lg1fa4ax480/h?rlkey=5d4ykq5br3kzkwarhi4ld4na8&dl=0)

<p align="center"><img src="./figure_pdfs/fig3.pdf" alt="disentanglement_example" style="display: block; margin-left: auto; margin-right: auto; width: 100%; text-align: center;" > </p>

#### Run

Use `python main.py <param-name> <param-value>`. For example:

```
python main.py --sup_signal continuousprice -s 1 --name continuousprice_s1 --btcvae-B 1 --btcvae-M 16
```

#### Output

This will create a directory `results/<model-name>/` which will contain:

* **specs.json**: The parameters used to run the program (default and modified with CLI).
* **train_losses.csv**: All (sub-)losses computed during training on the train and validation dataset.
* **test_losses.log**: All (sub-)losses computed at the end of training on the test1 and test2 dataset. 
* **reconstruct_traverse.png**: latent traversals of latent dimensions. 
* **filename_test1.csv**: filenames of all watches in the test1 dataset. 
* **filename_test2.csv**: filenames of all watches in the test2 dataset. 
* **filename_train.csv**: filenames of all watches in the train dataset. 
* **mean_params_test1.csv**: mean visual characteristics of all watches in the test1 dataset. 
* **mean_params_test2.csv**: mean visual characteristics of all watches in the test2 dataset. 
* **mean_params_train.csv**: mean visual characteristics of all watches in the train dataset. 

#### Help

```
usage: main.py ...

General options:
  name Name of the model for storing or loading purposes.
  -s, --seed SEED Random seed.
  -e, --epochs EPOCHS Maximum number of epochs to run for.
  -b, --batch-size BATCH_SIZE Batch size for training.
    --lr LR Learning rate.
  -z, --latent-dim LATENT_DIM Dimension of the latent variable.
  —tv --threshold-val Threshold for Masking.
  --sup_signal Choice of Signal. (Examples: discreteprice, brand, circa, material, movement or a combination in the format <signal1>_<signal2> etc.)

Loss specific hyperparameters:
  --btcvae-A BTCVAE_A			Weight of the MI term (alpha in the paper).
  --btcvae-G BTCVAE_G			Weight of the dim-wise KL term (gamma in the paper).
  --btcvae-B BTCVAE_B			Weight of the TC term (beta in the paper).
  --btcvae-M BTCVAE_M			Weight of the supervised loss term (delta in the paper).

```

#### Hyperparameter Selection

Run the same model configuration (i.e. a combination of the supervisory signal and loss-specific hyperparameters) for different random seeds. Select the optimal loss-specific hyperparameters for a particular supervisory signal combination based on the lowest supervised loss on the validation dataset (averaged across different seeds).


#### UDR Calculation

Run the most optimal loss-specific hyperparameters for each combination of the supervisory signal across different random seeds. Copy the files stored in ‘results/<model-name>/‘ directory with the filename ending in ‘mean_params_test2.csv’ to the ‘calculate_udr’ folder. Switch to an R environment and calculate UDR using the below command. You may need to modify the R script and adjust the model name.

```
Rscript udr_calculation.R --sup_signal='brand'
```

#### Conjoint Analysis and "Ideal Point'' Generative Design 

See the directory `/conjoint_analysis_and_ideal_point_design` for an example ipython notebook for running the hiearachical Bayeisan estimation and replicate results.

Note that generating the ideal point design requires inputting the "ideal point" embedding values into the generative model from the disentanglement portion of this codebase.

#### Steps to Reproduce Figures & Tables

1. Change directory to post_model_search folder and run the following command on a 64G GPU. 

```
python main.py --sup_signal brand_circa_movement -s 9 --name brand_circa_movement_s9 --btcvae-B 50 --btcvae-M 1
```

2. This will create a directory `results/brand_circa_movement_s9/` which will contain:

* **specs.json**: The parameters used to run the program (default and modified with CLI).
* **train_losses.csv**: All (sub-)losses computed during training on the train and validation dataset.
* **test_losses.log**: All (sub-)losses computed at the end of training on the test1 and test2 dataset. 
* **reconstruct_traverse.png**: latent traversals of latent dimensions. 
* **filename_test1.csv**: filenames of all watches in the test1 dataset. 
* **filename_test2.csv**: filenames of all watches in the test2 dataset. 
* **filename_train.csv**: filenames of all watches in the train dataset. 
* **mean_params_test1.csv**: mean visual characteristics of all watches in the test1 dataset. 
* **mean_params_test2.csv**: mean visual characteristics of all watches in the test2 dataset. 
* **mean_params_train.csv**: mean visual characteristics of all watches in the train dataset. 

Files named **reconstruct_traverse.png** will produce images similar to Figure 5 in the paper.

3. Change directory to figure_pdfs to see Figure 1-4.

4. Change directory to Figures+Tables and execute the figures_and_tables.Rmd to produce Table 2 and Figure 6 in the paper.

5. Change directory to conjoint_analysis_and_ideal_point_design and execute Example_Python_Notebook_with_Results_and_Plots.ipynb to produce Table 6-7 and Figure 10-12 in the paper. Note, that we would only get the visual characteristic vector corresponding to Figure 11 and would then have to pass it through the decoder of the disentanglement model to produce the designs seen in Figure 11. 

## List of Files

./README.md: README file
./disentanglement_env.yml: Environment File


./calculate_udr/calc_udr.csv: Sample Output of udr_calculation.R
./calculate_udr/udr_calculation.R: Script to Calculate UDR

./conjoint_analysis_and_ideal_point_design/conjoint_analysis_benchmark_models/data_generated.py: Data Generation for Conjoint
./conjoint_analysis_and_ideal_point_design/conjoint_analysis_benchmark_models/models.py: Prediction Models
./conjoint_analysis_and_ideal_point_design/conjoint_analysis_benchmark_models/requirements_benchmark_models.yml: Environment File
./conjoint_analysis_and_ideal_point_design/conjoint_analysis_benchmark_models/train_generated_watch_prediction.py: Watch Prediction Training Functions
./conjoint_analysis_and_ideal_point_design/data_conjointanalysis/CBCwatchexercise_simple_responses_generated.csv: Conjoint Survey Response CSV File
./conjoint_analysis_and_ideal_point_design/data_conjointanalysis/conj_gen_file_mapping_AnkitThresholds.csv: CSV file to map to actual visual characteristic quantified levels
./conjoint_analysis_and_ideal_point_design/data_conjointanalysis/FullConjointData_generated_mapped_variables.csv: Conjoint Survey Raw File 
./conjoint_analysis_and_ideal_point_design/data_conjointanalysis/disentanglement_example.png: Disentanglement Example
./conjoint_analysis_and_ideal_point_design/Example_Python_Notebook_with_Results_and_Plots.ipynb: Conjoint Interactive Python Notebook
./conjoint_analysis_and_ideal_point_design/data.py: Data Generation for Conjoint 
./conjoint_analysis_and_ideal_point_design/HB_conjoint_requirements.yml: Environment File
./conjoint_analysis_and_ideal_point_design/run_HB_conjoint.py: Hierarchical Bayesian Conjoint Prediction File

./hyperparameter_selection/__pycache__/*: bytecode cache files automatically generated by python

./hyperparameter_selection/data/watches/christies.npz: Training Data
./hyperparameter_selection/data/watches/christies_test1.npz: Test1 Data
./hyperparameter_selection/data/watches/christies_test2.npz: Test2 Data

./hyperparameter_selection/dataset/__pycache__/*: bytecode cache files automatically generated by python
./hyperparameter_selection/dataset/datasets.py: for processing data

./hyperparameter_selection/models/__pycache__/*: bytecode cache files automatically generated by python
./hyperparameter_selection/models/initialization.py: initializing the neural network
./hyperparameter_selection/models/losses.py: computing the neural network losses
./hyperparameter_selection/models/math.py: helper file with useful math functions
./hyperparameter_selection/models/modelIO.py: helper file for reading/writing model
./hyperparameter_selection/models/regression.py: supervised layer
./hyperparameter_selection/models/vae.py: code for setting up the VAE

./hyperparameter_selection/training/__pycache__/*: bytecode cache files automatically generated by python
./hyperparameter_selection/training/evaluate.py: code to evaluate the trained model
./hyperparameter_selection/training/training.py: code to train the model

./hyperparameter_selection/utils/__pycache__/*: bytecode cache files automatically generated by python
./hyperparameter_selection/utils/__init__.py
./hyperparameter_selection/utils/helpers.py: helper functions
./hyperparameter_selection/utils/visualize.py: code to visualize the learned visual characteristics
./hyperparameter_selection/utils/viz_helpers.py: helper functions for visualization

./hyperparameter_selection/hyperparam.ini: configuration file for hyperparameters
./hyperparameter_selection/example_commands.txt: file with example commands

./hyperparameter_selection/main.py: main python execution file 



./post_model_search/__pycache__/*: bytecode cache files automatically generated by python

./post_model_search/data/watches/christies.npz: Training Data
./post_model_search/data/watches/christies_test1.npz: Test1 Data
./post_model_search/data/watches/christies_test2.npz: Test2 Data

./post_model_search/dataset/__pycache__/*: bytecode cache files automatically generated by python
./post_model_search/dataset/datasets.py: for processing data

./post_model_search/models/__pycache__/*: bytecode cache files automatically generated by python
./post_model_search/models/initialization.py: initializing the neural network
./post_model_search/models/losses.py: computing the neural network losses
./post_model_search/models/math.py: helper file with useful math functions
./post_model_search/models/modelIO.py: helper file for reading/writing model
./post_model_search/models/regression.py: supervised layer
./post_model_search/models/vae.py: code for setting up the VAE

./post_model_search/training/__pycache__/*: bytecode cache files automatically generated by python
./post_model_search/training/evaluate.py: code to evaluate the trained model
./post_model_search/training/training.py: code to train the model

./post_model_search/utils/__pycache__/*: bytecode cache files automatically generated by python
./post_model_search/utils/__init__.py
./post_model_search/utils/helpers.py: helper functions
./post_model_search/utils/visualize.py: code to visualize the learned visual characteristics
./post_model_search/utils/viz_helpers.py: helper functions for visualization

./post_model_search/hyperparam.ini: configuration file for hyperparameters
./post_model_search/example_commands.txt: file with example commands

./post_model_search/main.py: main python execution file 
./post_model_search/main_viz.py: main python execution file to generate images for conjoint analysis

./figure_pdfs/fig1.pdf: Figure 1 of the Paper
./figure_pdfs/fig2.pdf: Figure 2 of the Paper
./figure_pdfs/fig3.pdf: Figure 3 of the Paper
./figure_pdfs/fig4.pdf: Figure 4 of the Paper

## Computing Resources

We used 64G GPU to run each model instance of 100 epoch, 64 batch size, 5e-4 learning rate. Each model run takes 8 minutes. We train 10 random seeds * 32 supervisory signal combinations * 25*15 grid values = 120,000 models. This would mean 16000 hours of model training.

## Citation

Feel free to use this code for your research. If you find this code useful, please use the following citation information in plaintext or BibTeX format:

Sisodia, A, Burnap, A, and Kumar, V (2024). Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis. Journal of Marketing Research (under review).

```
@article{sisodia2024disentangelment,
  title= Generative Interpretable Visual Design: Using Disentanglement for Visual Conjoint Analysis},
  author={Sisodia, A, Burnap, A, and Kumar, V},
  journal={Journal of Marketing Research (under review)},
  volume={XXX},
  number={XXX},
  pages={XXX},
  year={2024},
  publisher={XXX}
}
```

## Acknowledgments
Portions of this codebase were built on elements from the following open source projects, please see their respective licenses and cite accordingly:
* [disentangling-vae](https://github.com/YannDubs/disentangling-vae)
