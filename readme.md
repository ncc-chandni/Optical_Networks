# Optical Network QoT Estimation using Machine Learning

## Project Overview

This project is inspired by the research papers *"Active Wavelength Load as a Feature for QoT Estimation Based on Support Vector Machine"* and *"A Performance Analysis of Supervised Learning Classifiers for QoT Estimation in ROADM-based Networks"*. The primary focus of this project is to measure the Quality of Transmission (QoT) using the Optical Signal-to-Noise Ratio (OSNR) metric and to apply a Support Vector Machine (SVM) model for predicting QoT. Additionally, the project explores and compares the performance of various machine learning models, using the F1 score as a key evaluation metric.

## Project Structure

- **Files Folder**: Contains the referenced research papers and their summaries, providing theoretical background and context for the project.

- **`file_reader.py`**: A script for reading the dataset and processing it into structured classes and numpy array for subsequent analysis and modeling.

- **`plot_statistics.py`**: A script used to plot the Receiver Operating Characteristic (ROC) curve and Confusion Matrix, providing visual insights into model performance.

- **`machine_learning_optical_network.ipynb`**: A Jupyter notebook where the SVM model is implemented to make predictions on the QoT in terms of the OSNR Optical Signal to Noise Ratio of an optical lightpath. . The notebook also measures the model's accuracy and evaluates its performance.

- **`comparing_algorithms.ipynb`**: This notebook compares the performance of various machine learning models in predicting QoT, with a focus on the F1 score as a performance metric.

## Additional Notes

- The project references a GitHub repository by [adiazmont](https://github.com/adiazmont/machine-learning-for-optical-networks/tree/master), which served as a valuable resource during development.
- The analysis performed in this project offers insights into the application of machine learning in optical networks, a crucial area for optimizing modern communication systems.

