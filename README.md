# Molecular Property Prediction with LLMs

This repository contains code for predicting molecular properties directly from SMILES strings using Large Language Models (LLMs) from the Hugging Face Transformers library. This approach leverages the ability of LLMs to learn complex patterns from sequence data to predict properties like solubility, toxicity, and other relevant chemical characteristics.

## Project Overview

Predicting molecular properties is crucial in various fields, including drug discovery, materials science, and environmental science. Traditional methods often involve computationally expensive simulations or complex machine learning models trained on engineered molecular features. This project explores a more recent approach: using LLMs to directly analyze SMILES strings, a textual representation of molecular structures.

## Key Features

*   Uses pre-trained LLMs from Hugging Face Transformers.
*   Supports regression tasks (predicting continuous properties).
*   Provides options for loading data from a CSV file or using datasets from MoleculeNet.
*   Includes training, evaluation, and prediction functionalities.
*   Includes example of how to use the trained model to predict the property of any molecule.

## Usage

1.  **Prepare your data:**

    *   **Option 1: CSV file:** Create a CSV file named `your_data.csv` with columns named "SMILES" and "Property".
    *   **Option 2: MoleculeNet:** The code also supports loading datasets from MoleculeNet directly (e.g., Delaney).

2.  **Run the training script:**

    ```bash
    python your_script_name.py #replace your_script_name with the name of your python script
    ```

    This will train the LLM on your data and save the trained model and tokenizer to a directory named `my_molecular_property_model`.

3.  **Predicting the property of a molecule:**

After training the model, you can use the `predict_property` function to predict the property of any molecule. Here's how:

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

#Load the model and tokenizer
model_name = "my_molecular_property_model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def predict_property(smiles_string):
    #... (the function as defined in the code)

new_smiles = "CCOC(=O)C"  # Example: Ethyl acetate
predicted_property = predict_property(new_smiles)
print(f"Predicted property for {new_smiles}: {predicted_property}")
