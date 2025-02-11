import copy
import concurrent.futures as cfuts
import json
from tqdm import tqdm
from src import utils, model

class BaseGenerator:
    def __init__(self, dataset_name, model_name, technique_name, args):
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.technique_name = technique_name
        self.args = args
        self.system_message = 'Only generate the Python code.'

    def form_technique_prompt(self, prompt):
        """
        This method should be overridden to generate the prompt for the specific technique.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def generate_prompt(self, dataset):
        """
        This method should be overridden to generate the prompt for the specific technique.
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def generate_result(self, messages, data):
        """
        This method should be overridden to implement dataset-specific result generation.
        """
        raise NotImplementedError("Subclasses must implement this method.")
    
    def run_model(model_name):
        """
        This method should be overridden to run the model with the generated prompt.
        """
        raise NotImplementedError("Subclasses must implement this method.")

