# project_network_science

## Experimental Setup
This project contains the code required to reproduce the experiments conducted in the [comparative analysis of different models](./paper/project_network_science.pdf) for the Network Science lecture. The code contains an implementation of the Louvain-Core algorithm as well as implementations for the NMI metric and the Map-Equation.

## Download code
The code will be available as a copy within this zip. However, it is advised to download the most recent code from github.
```console
git clone https://github.com/Olu93/project_network_science.git
cd network_science
```

## Installation
In order to run the experiment, you have to clone and run the jupyter notebook experiment.ipynb. Make sure you have conda or python installed. Then install the requirements.txt dependencies.

```console
pip install -r requirements.txt
```

If this installation fails, the directory also includes the environment.yml file to install an environment via conda.

```console
conda env create -f environment.yml
conda activate ptgpu
```


## Running the experiment
After installing the requirements, you'll be able to run the experiment. The code will run for a couple of hours. Furthermore, it will use some of your computational ressources, as it uses multiprocessing to speed up the computation. The number of cores that will be used can be easily tweaked in experiment.py.

```console
python experiment.py
```

## Running the data processing
After the experiment completed, you will find a number of csv in the folder. By putting all of them in the data folder and running Jupyter, you'll have the opportunity to process the data yourself. The data_processing_and_visualisation.ipynb provides the code to aggregate all csv's and provides some basic graphs. To run jupyter you will have to run the command:
```console
jupyter notebook
```
Jupyter will open. Navigate to ./src/data_processing_and_visualisation.ipynb and open the file. Run all the cells within Jupyter notebook.  After completion you will see the plots that were mentioned in the paper. Make sure that all the data files are in the ./data folder. Don't include any unnecessary files to it.


## File structure
A brief listing of the relevant file structure for this code. All the relevant code lies in **src**.
- **algorithms:** # Contains all the algorithms, including Louvain Core and it's specializations. Also includes the MapEquation Algorithm
- **data:** # Contains the current data of that was referenced in the paper
- **dump:** # Mostly a bunch of files that were used during development. Of no particular interest.
- **helper:** # Some of the helper functions used throughout the code
- **playgrounds:** # Jupyter like code scripts to aid the development. They are of nor particluar interest

