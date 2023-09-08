# Modeling Complex Disease Trajectories using Deep Generative Models with Semi-Supervised Latent Processes

This repository contains the code for the paper *Modeling Complex Disease Trajectories using Deep Generative
Models with Semi-Supervised Latent Processes* under review for <cite>[ML4H 2023][1]</cite>. 

## Data

The EUSTAR data used to produce the results in the paper is confidential and cannot be shared. However, to facilitate reproducibility, we implemented a framework that allows the user to generate artificial data with the same structure as EUSTAR. The artificial data is randomly generated, and thus the evaluation results on this artificial data do not reflect the evaluation results presented in the paper.

## Code 
The model implementation builds upon the 
<cite>[pythae][2]</cite> library developed by Chadebec & al. The **benchmark_VAE/src/pythae/** folder contains the model implementation and the framework specific to systemic sclerosis modeling in the **benchmark_VAE/src/pythae/ssc/** folder. The **fake_data/** folder contains the framework to generate the artificial data. The **demo_notebooks/** folder contains notebooks to experiment with the model on some artificial data.

 ## Running experiments on artificial data
 
 Please be mindful to adapt the paths to save, load data, models, figures etc. to match your environment (`path_to_project` variable at the beginning of the scripts/notebooks)
1.  `pip install -r requirements.txt`
2.  `python3 fake_data/generate_fake_data.py` to generate fake data (then stored in fake_data/raw)
3. `python3 benchmarkVAE/src/pythae/ssc/create_cv.py` to create the cohorts and objects (takes some time depending on the size of the data)

After succesful completion of these three steps, you can run the notebooks in **demo_notebooks/**.


### Notebooks description

* **demo_notebooks/a_train_model.ipynb** : train and save a model. This notebook has to be executed first, as the other notebooks will reload the saved trained model.

* **demo_notebooks/clustering.ipynb** : latent space clustering and trajectory similarity

* **demo_notebooks/latent_space.ipynb** : latent space visualizations 

* **demo_notebooks/model_evaluation.ipynb** : online monitoring and performance evaluation


[1]: https://ml4health.github.io/2023/
[2]: https://pypi.org/project/pythae/