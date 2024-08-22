import numpy as np
import pandas as pd 
import itertools
from itertools import cycle 
import matplotlib.pyplot as plt 
from scipy import interpolate 
from sklearn.metrics import auc 
import matplotlib.font_manager as font_manager 

fontsize_ax = 12 
font_name = 'Times New Roman'
font = font_manager.FontProperties(family=font_name, size= fontsize_ax)

class PlotStats():
    def __init__(self):
        plt.figure()
        plt.rcParams['figure.figsize'] = [10,8]

# Key Components of the Function:
# Parameters:

# cm: The confusion matrix to be plotted.
# classes: The list of class labels (not used directly in this code).
# normalize (default: False): If True, the confusion matrix will be normalized (i.e., each value will be divided by the sum of values in the corresponding row).
# title (default: 'Confusion matrix'): The title of the plot (commented out in this code).
# cmap (default: plt.cm.Blues): The color map used to display the matrix.

# Normalization:
# If normalize is set to True, the confusion matrix values are normalized by dividing each row by its sum, 
# turning the values into proportions rather than raw counts.

# Matrix Visualization:
# The matrix is visualized using plt.imshow with the specified color map (cmap).
# tick_marks: This is where the class labels are set on the x and y axes. In this code, it's set to 
# display labels ['16QAM', '8QAM', 'QPSK', 'None'].

# Text Annotations:
# The code iterates over all cells in the matrix (i, j) and places a text annotation in each cell displaying 
# the corresponding value. The text color changes depending on whether the cell's value is above or below half 
# of the maximum value in the matrix (thresh).

# Axes Labels:
# Labels for the x-axis (Predicted label) and y-axis (True label) are added.

# Final Display:
# plt.tight_layout() is called to ensure the layout fits well in the figure, followed by plt.show() to display the plot.

    def plot_confusion_matrix(self, cm, classes, normalize=False, title='Confusion Matrix', cmap = plt.cm.Blues):
        if normalize:
            cm = cm.astype('float')/cm.sum(axis=1)[:, np.newaxis]
            print("Normalized Confusion Matrix")
        else:
            print("Confusion Matrix, without normlization")
        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        tick_marks = np.arange(len(classes))
        list_ = ['16QAM', '8QAM', 'QPSK', 'None', ]
        plt.xticks(tick_marks, fontsize = 10, rotation=45, fontname = font_name)
        plt.yticks(tick_marks, list_, fontsize= 10, fontname = font_name)

        fm = '.2f' if normalize else 'd'
        thresh = cm.max() / 2. 
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fm),
                     horizontalalignment='center', fontsize=fontsize_ax, fontname = font_name,
                     color='white' if cm[i,j] > thresh else "black")
            
        plt.xlabel("Predicted label", fontsize=fontsize_ax, fontname = font_name)
        plt.ylabel("True label", fontsize=fontsize_ax, fontname= font_name)
        plt.tight_layout()

        plt.show()



    def plot_roc_curve(self, fpr, tpr, roc_auc):
        lw = 2 
        plt.plot(fpr, tpr, color = 'darkorange', lw=lw, label='ROC curve (area= %0.2f)'% roc_auc) 
        plt.plot([0,1], [0,1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title("Receiver operating characteristic of QoT-E (1)")
        plt.legend(loc='lower right')
        plt.show()

# The plot_roc_curve_multi function you've shared is designed to plot the ROC (Receiver Operating Characteristic) curves 
# for a multi-class classification problem. This function aggregates ROC curves across multiple classes and plots them, 
# including both micro-averaged and macro-averaged ROC curves. Here's a breakdown of what each part of the code does:

# Breakdown of the Code
# Aggregate False Positive Rates (FPR):
# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)])): This line creates a combined array of unique
# false positive rates (fpr) from all classes.

# Interpolate True Positive Rates (TPR):
# mean_tpr = np.zeros_like(all_fpr): This initializes an array to store the mean true positive rate (tpr) across all classes.
# The loop for i in range(n_classes): iterates over each class to calculate the interpolated tpr values at each fpr point.

# Calculate Mean TPR and Macro AUC:
# mean_tpr /= n_classes: Averages the tpr values across all classes.
# roc_auc["macro"] = auc(fpr["macro"], tpr["macro"]): Computes the Area Under the Curve (AUC) for the macro-averaged ROC curve.

# Plotting the ROC Curves:
# The function plots different ROC curves, including:
# Micro-average ROC curve: Combines the true positives and false positives across all classes and calculates the ROC.
# Macro-average ROC curve: Averages the ROC curves of all classes.
# Class-specific ROC curves: Plots the ROC curve for each specific class (e.g., for OSNR thresholds like ≥17 dB, ≥14 dB, etc.).
# The diagonal line (plt.plot([0, 1], [0, 1], 'k--', lw=lw)) represents a random classifier with an AUC of 0.5.

# Final Customization:
# plt.xlabel, plt.ylabel, plt.xticks, plt.yticks: Customizes the labels and ticks for the axes.
# plt.title('Receiver operating characteristic to multi-class QoT-E'): Sets the title of the plot.
# plt.legend: Adds a legend to identify each curve.

    def plot_roc_curve_multi(self, fpr, tpr, roc_auc, n_classes):
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(n_classes):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

        mean_tpr /= n_classes

        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

        plt.plot(fpr["micro"], tpr["micro"], 
                label="micro-average ROC curve (AUC= {0:0.2f})"
                "".format(roc_auc['micro']),
                color='deeppink', linestyle=":", linewidth=4)
        
        plt.plot(fpr["macro"], tpr["macro"], 
        label="macro-average ROC curve (AUC= {0:0.2f})"
        "".format(roc_auc['macro']),
        color='navy', linestyle=":", linewidth=4)

        lw=2 
        plt.plot(fpr[0], tpr[0], color='turquoise', lw=lw,
                 label='ROC curve of OSNR >= 17 dB (AUC = {1:0.02f})'
                 ''.format(0, roc_auc[0]))
        plt.plot(fpr[1], tpr[1], color='lightyellow', lw=lw,
                 label='ROC curve of OSNR >= 14 dB (AUC = {1:0.02f})'
                 ''.format(1, roc_auc[1]))
        plt.plot(fpr[2], tpr[2], color='red', lw=lw,
                 label='ROC curve of OSNR >= 10 dB (AUC = {1:0.02f})'
                 ''.format(2, roc_auc[2]))
        plt.plot(fpr[3], tpr[3], color='darkviolet', lw=lw,
                 label='ROC curve of OSNR < 10 dB (AUC = {1:0.02f})'
                 ''.format(3, roc_auc[3]))
        
        plt.plot([0,1], [0,1], 'k--', lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xticks(fontsize=fontsize_ax, fontname = font_name)
        plt.yticks(fontsize=fontsize_ax, fontname = font_name)
        plt.xlabel("False Positive Rate", fontsize=fontsize_ax, fontname=font_name)
        plt.ylabel("True Positive Rate", fontsize = fontsize_ax, fontname = font_name)
        plt.title('Receiver operating characteristic to multi-class QoT-E')
        plt.legend(loc="lower right", prop=font, fontsize=10)
        plt.show()