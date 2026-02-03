import pickle
from csv import reader
from models import *
import pandas as pd

"""
PV optimization on DNS dataset of Xu using an encoder-decoder architecture (Kamila Zdybal)
Version: Load different datatypes
Author: Grégoire Corlùy (gregoire.stephane.corluy@ulb.be)
Date: October 2024
Python version: 3.10.10
"""

class loadData:
    """ Load model, curves and metadata """

    def __init__(self, filename_state_space_names, path_metadata, filename_metadata):
        
        self.path_metadata = path_metadata
        self.filename_metadata = filename_metadata
        self.metadata = self.loadMetadata()

        self.filename_state_space_names = filename_state_space_names
        self.filename_model = self.metadata["model_name"]
        self.filename_curve = self.metadata["curve_name"]

        self.model_params = self.metadata["model_params"]

    def loadStateSpaceNames(self, path_data):
        """Returns the list of species names contained in the dataset.

        Args:
            path_data (str): Path where the filename containing the state space names is.

        Returns:
            list[str]: List with the name of all the species contained in the state space.
        """
        
        with open(f"{path_data}{self.filename_state_space_names}", mode='r', newline='') as file:
            csv_reader = reader(file)
            list_species = [row[0] for row in csv_reader]

        return list_species

    def loadMetadata(self):
        """Load a dictionary containing all the metadata.

        Returns:
            dict[str, Any]: Dictionary containing the metadata of the trained encoder-decoder.
        """

        with open(f'{self.path_metadata}/{self.filename_metadata}', 'rb') as f:
            loaded_dict = pickle.load(f)

        return loaded_dict
    
    def loadModel(self, filename_model = None):
        """Load the encoder-decoder model.

        Args:
            filename_model (str, optional): Name of the trained encoder-decoder. Defaults to None.

        Returns:
            PV_autoencoder: Trained encoder-decoder.
        """

        if(filename_model is None):
            filename_model = self.filename_model

        model_reloaded = PV_autoencoder(**self.model_params)
        model_reloaded.load_state_dict(torch.load("out/" + filename_model, weights_only=False))

        return model_reloaded

    def loadCurves(self):
        """Returns the arrays of the training and validation losses from the training of the encoder-decoder, being saved in a csv file.

        Returns:
            (pandas.Series, pandas.Series): 
                A tuple with:
                - training: List of training losses for every epoch.
                - validation: List of validation losses for every epoch.
        """

        df = pd.read_csv('curves/' + self.filename_curve, header=None)

        training = df.iloc[0]  #Training curve
        validation = df.iloc[1]  #Validation curve

        return training, validation

    def getInputSpecies(self, path_data):
        """Returns the list of input species names given the filename containing the species names.

        Args:
            path_data (str): Name of the csv file containing the species names.

        Returns:
            list[str]: List of input species names.
        """

        list_species = self.loadStateSpaceNames(path_data)
        idx_remove = self.metadata["idx species removed"]

        for index in sorted(idx_remove, reverse=True):
            del list_species[index]
        
        return list_species
