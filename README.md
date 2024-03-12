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
This code will reproduce Figures (5, 6, 10, 12, G.1, G.2) and Tables (2, 6, 7, D.1, E.1, F.1). Some of the other figures (1, 2, 3, 4) from the paper are at './figure_pdfs'. 

## Installation
#### Libraries
This codebase was built using the Pytorch neural net framework.  However, it also depends on over 100 other Python packages, as well as GPU libraries including CUDA runtimes and cuDNN kernel support. We can not guarantee this code will run and replicate unless the same environment and library versions are used.  To install these packages (including the same library versions), please use the following:

    conda env create -f disentanglement_env.yml

#### Data
To download the dataset, please use the following links. Copy these files to ‘./hyperparameter_selection/data/watches/‘ and ‘./post_model_search/data/watches/‘ directories.

* #### [Watches Dataset](https://www.dropbox.com/scl/fo/akj3w8pat0lg1fa4ax480/h?rlkey=5d4ykq5br3kzkwarhi4ld4na8&dl=0)

## Replication Steps

#### Step 1

Go to './hyperparameter_selection' and run disentanglement model with a unique $\beta$, $\delta$, and supervisory signal combination with 10 different seeds. Vary $\beta$, $\delta$, and supervisory signal combination.

For example, in the below command, the seed is set to 1, $\beta$=18, $\delta$=50, and the supervisory signal is brand. The model name is brand_s1. 

```
python main.py --sup_signal brand -s 1 --name brand_s1 --btcvae-B 18 --btcvae-M 50
```

In the above command, seed, $\beta$, and $\delta$ is a scalar value. This codebase, specific to the watch dataset, supports the following set of discrete supporting signals. Using any other name will result in an error.

```
discreteprice
brand
circa
material
movement
discreteprice_brand
discreteprice_circa
discreteprice_material
discreteprice_movement
brand_circa
brand_material
brand_movement
circa_material
circa_movement
material_movement
discreteprice_brand_circa
discreteprice_brand_material
discreteprice_brand_movement
discreteprice_circa_material
discreteprice_circa_movement
discreteprice_material_movement
brand_circa_material
brand_circa_movement
brand_material_movement
circa_material_movement
discreteprice_brand_circa_material
discreteprice_brand_circa_movement
discreteprice_brand_material_movement
discreteprice_circa_material_movement
brand_circa_material_movement
discreteprice_brand_circa_material_movement
```

The above command will create a directory `results/<model-name>/` which will contain:

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

Select the value of $\beta$ and $\delta$ for each supervisory signal at which the average supervised loss across 10 seeds on a validation dataset is lowest. The supervised loss on the test set is stored as 'sup_loss_test' in the json file with the name ending in test_losses.log in the directory `results/<model-name>/` for each combination of seed, $\beta$, $\delta$, and the supervisory signal

#### Step 2

Go to './post_model_search' and run disentanglement model at the optimal $\beta$ and $\delta$ for each supervisory signal combination at 10 different seeds. 

For the watch dataset, execute the commands listed in execute_step2.txt to use the values listed in the paper. For example, see the first command in the file 'execute_step2.txt':

```
python main.py --sup_signal continuousprice -s 1 --name continuousprice_s1 --btcvae-B 1 --btcvae-M 16 
```

The above command will create a directory `results/<model-name>/` which will contain:

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

Copy the files stored in ‘results/<model-name>/‘ directory with the filename ending in ‘mean_params_test2.csv’ to the ‘calculate_udr’ folder.

#### Step 3

Go to './calculate_udr' to compare the UDRs for different supervisory signals. Switch to an R environment and execute the Rscript udr_calculation.R with the supervisory signal as the argument. For example:

```
Rscript udr_calculation.R --sup_signal='brand'
```

The results will be appended to filenamed udr.log. It will replicate results in Table F.1 of the paper.

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
