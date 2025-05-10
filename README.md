# Earnings Call Transcript Analysis

### Structure
The repository is split into two sections, data and models. Data contains all files related to data for our project, and models contains all files related to running the models.
### Code
There is a folder that contains all of the code that runs our different models and the output files from running them.
### Data
There is a folder for the data which contains folders that contain the earnings call transcripts, labels for the data, and baseline data.
### Running the code
Running the code is tedious. The aptly named models folder contains code for each model.  The BOW_models.py, train_short_context.py, and train_bigbird.py are self-contained python codes that are ready to run assuming you change the file path of the data.  For the hierarchical transformer and finbert codes there are jupyter notebooks that have been optimized for use on Google Colab, since they required more processing power to train.  In order to use them, you clone the repository into Colab by first generating an SSH key on the Colab runtime, then copying it to your GitHub account.  You must also link your Google Drive and ensure it has enough free space available, since that is where the code saves the trained models to.
### Link to data source
https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/TJE0D0
https://paperswithcode.com/dataset/earnings-call

### Models
