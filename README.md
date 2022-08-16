# Laser Projekt Machine Learning 1

Author: Alexander Pieper

This is a Machine Learning Project, where we are building a binary classification model for medical lasers.
This Code exists in two versions:
1. in a Jupyter Notebook, where all the content is explained and commented
2. in a Python Module with 5 `.py` scripts 

The main contents and results of this Project are probably best understood in the Jupyter Notebook `Code/project_notebook.ipynb`, as there are many descriptions.
The Python scripts in the Directory `Code/` are better for Development though. 
Here one can add new Models, that should be benchmarked against the others quite quickly. 
Have a look at the commit history, as there are some examples of how to do that.


## How to run the `Code/main.py` pipeline in a terminal:
- Clone this repository into your local machine
- Open a Terminal and cd into this directory
- install all the dependencies with `pip3 install -r requirements.txt` (or `pip`, depending on your settings)
- run `export PYTHONPATH=.`, so that the submodules can be imported
- to run the pipeline, execute `python3 Code/main.py` (or `python`, depending on your settings)
    - it should then print some tables, followed by some printed evaluations of models.
    - at the end it should start a dashboard, saying `Dash is running on http://127.0.0.1:8050/`

## How to run the `Code/project_notebook.ipynb` in a jupyter notebook:
- Clone this repository into your local machine
- Just open that file in a Jupyter instance (or any other editor that supports Python notebooks)
- at the start, there are the installations of the necessary modules. 
After installing them you might need to restart the Kernel.
- Then just go down every cell. There are descriptions that describe what is happening and how to change things if you want to  



