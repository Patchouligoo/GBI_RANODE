import os
import importlib
import luigi
import law

from src.utils.law import BaseTask

class Preprocessing(BaseTask):

    def output(self):
        return law.LocalFileTarget("preprocessed_data.npy")
    
    def run(self):
        return