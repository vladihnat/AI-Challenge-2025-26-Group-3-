# Pollinator Detection – Starting Kit
--- 

## Overview

This notebook serves as a starting kit for the Pollinator detection project. Its purpose is to help users quickly understand how to load, explore, and manipulate the data, as well as how to begin building simple models using the provided tools.

The notebook is designed to be self-contained and easy to run, even for users discovering the project for the first time.

--- 

## Automatic Data Download

When executing the notebook, the required dataset is automatically downloaded if it is not already present on your system.
No manual setup is needed: as long as you run the cells in order, the notebook will check for the data locally and retrieve it if necessary. For quick experimentation, the script allows you to charge only a fraction of the dataset on the RAM to execute the rest of the cells.

---

## Dataset Structure

The data is stored in `.h5` files. Each file corresponds to a sequence of images, similar to a short stop-motion recording, and is associated with a single binary label indicating whether a pollinator is present or not.

---

## Exploratory Analysis

Several sections of the notebook are devoted to understanding the dataset. Visualizations are used to examine class distribution and highlight the strong imbalance between sequences with and without pollinators.

The notebook also includes tools to visualize individual frames and sequences, helping users develop an intuition for what pollinator presence looks like in the data.

Finally, it gives some hints about feature extracting using PCA.

---

## First Modeling Experiments

To help users get started, the notebook demonstrates simple baseline modeling ideas and shows how to evaluate predictions using metrics that are appropriate for imbalanced binary classification.

These examples are not meant to be optimal solutions, but rather a foundation on which users can build and experiment with their own approaches.

---

## Prepare Submission for Codabench

At the end of the notebook, a final cell shows how to generate a properly formatted submission for Codabench.

You should zip a folder containing your model.py with a class Model inside. You class should have a \_\_init\_\_ method, a fit one and a predict one. Also add a requirements.txt with all libraries needed to run your code. Everything is well explained in the starting kit, and the starting kit already contains a sample submission to help you understand the requirements.

Download starting kit here (or in the Files section of this challenge): https://www.codabench.org/datasets/download/2a54ab1f-0d83-4375-a0b4-93036ec2d904/

