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
