import numpy as np
import itertools, pickle

class Database:

    def __init__(self, file = None):
        
        if file != None:
            self.load(file)


    def save(self, file):
        """
        Saves the database in a .pkl file that can be loaded later.

        Parameters
        ----------
        file : str
            The name of the file to which the database will be saved.

        Returns
        -------
        None
        """
        with open(file, mode="wb") as opened_file:
            pickle.dump(self, opened_file)

    def load(self, file):
        """
        Loads the database from a .pkl file.

        Parameters
        ----------
        file : str
            The name of the file from which the database will be loaded.

        Returns
        -------
        None
        """
        with open(file, mode="rb") as opened_file:
            loaded_db = pickle.load(opened_file)
            return loaded_db