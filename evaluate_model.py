#!/usr/bin/env python

import os
import sys
import numpy as np
from helper_code import load_patient_data, get_shape, load_challenge_outputs, compare_strings
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score

# ✅ Function to find label and output files
def find_challenge_files(label_folder, output_folder):
    label_files, output_files = [], []
    for label_file in sorted(os.listdir(label_folder)):
        label_file_path = os.path.join(label_folder, label_file)
        if os.path.isfile(label_file_path) and label_file.lower().endswith('.txt'):
            root, _ = os.path.splitext(label_file)
            output_file_path = os.path.join(output_folder, root + '.csv')
            if os.path.isfile(output_file_path):
                label_files.append(label_file_path)
                output_files.append(output_file_path)
            else:
                print(f"⚠️ Warning: Missing output file for {label_file}")
    return label_files, output_files

# ✅ Function to load shape labels for "Diamond", "Plateau", "Decrescendo"
def load_shapes(label_files):
    valid_indices, labels = [], []
    included_labels = ["Diamond", "Plateau", "Decrescendo"]  # Defined within function to match original structure
    
    for i, file in enumerate(label_files):
        data = load_patient_data(file)
        label = get_shape(data)  # Extract murmur shape
        
        if label in included_labels:
            labels.append([int(label == "Diamond"), int(label == "Plateau"), int(label == "Decrescendo")])
            valid_indices.append(i)
    
    return np.array(labels, dtype=int), valid_indices

# ✅ Function to load classifier outputs for "Diamond", "Plateau", "Decrescendo"
def load_classifier_outputs(output_files, valid_indices):
    binary_outputs, scalar_outputs = [], []
    included_labels = ["Diamond", "Plateau", "Decrescendo"]  # Defined within function to match original structure
    
    filtered_output_files = [output_files[i] for i in valid_indices]
    
    for file in filtered_output_files:
        _, patient_classes, _, patient_scalar_outputs = load_challenge_outputs(file)
        
        binary_output = [0, 0, 0]  # Default (all classes 0)
        scalar_output = [0.0, 0.0, 0.0]  # Default probabilities
        
        for j, x in enumerate(included_labels):
            for k, y in enumerate(patient_classes):
                if compare_strings(x, y):
                    scalar_output[j] = patient_scalar_outputs[k]
                    binary_output[j] = int(patient_scalar_outputs[k] >= 0.5)  # Default threshold
        
        binary_outputs.append(binary_output)
        scalar_outputs.append(scalar_output)
    
    return np.array(binary_outputs, dtype=int), np.array(scalar_outputs, dtype=np.float64)

# ✅ Compute the best threshold using F1-score

# ✅ Compute evaluation metrics
# ✅ Compute evaluation metrics for "Diamond", "Plateau", "Decrescendo"
def compute_auc(labels, outputs):
    try:
        auroc_diamond = roc_auc_score(labels[:, 0], outputs[:, 0])
        auprc_diamond = average_precision_score(labels[:, 0], outputs[:, 0])

        auroc_plateau = roc_auc_score(labels[:, 1], outputs[:, 1])
        auprc_plateau = average_precision_score(labels[:, 1], outputs[:, 1])

        auroc_decrescendo = roc_auc_score(labels[:, 2], outputs[:, 2])
        auprc_decrescendo = average_precision_score(labels[:, 2], outputs[:, 2])
    except ValueError:
        auroc_diamond, auprc_diamond = 0.5, 0.5
        auroc_plateau, auprc_plateau = 0.5, 0.5
        auroc_decrescendo, auprc_decrescendo = 0.5, 0.5

    return (auroc_diamond, auprc_diamond, auroc_plateau, auprc_plateau, auroc_decrescendo, auprc_decrescendo)

# ✅ Compute F-measure (F1-score) for "Diamond", "Plateau", "Decrescendo"
def compute_f_measure(labels, outputs):
    f1_diamond = f1_score(labels[:, 0], outputs[:, 0])
    f1_plateau = f1_score(labels[:, 1], outputs[:, 1])
    f1_decrescendo = f1_score(labels[:, 2], outputs[:, 2])

    return np.mean([f1_diamond, f1_plateau, f1_decrescendo]), [f1_diamond, f1_plateau, f1_decrescendo]

# ✅ Compute accuracy for "Diamond", "Plateau", "Decrescendo"
def compute_accuracy(labels, outputs):
    accuracy_diamond = accuracy_score(labels[:, 0], outputs[:, 0])
    accuracy_plateau = accuracy_score(labels[:, 1], outputs[:, 1])
    accuracy_decrescendo = accuracy_score(labels[:, 2], outputs[:, 2])

    return np.mean([accuracy_diamond, accuracy_plateau, accuracy_decrescendo]), [accuracy_diamond, accuracy_plateau, accuracy_decrescendo]

# ✅ Compute weighted accuracy for "Diamond", "Plateau", "Decrescendo"
def compute_weighted_accuracy(labels, outputs):
    # Define a custom weight matrix for three classes
    weights = np.array([
        [5, 2, 1],  # Diamond
        [2, 5, 1],  # Plateau
        [1, 2, 5]   # Decrescendo
    ])

    # Initialize the confusion matrix for three classes
    confusion = np.zeros((3, 3))

    for i in range(len(labels)):
        true_class = np.argmax(labels[i])   # Find the true class index
        pred_class = np.argmax(outputs[i])  # Find the predicted class index
        confusion[pred_class, true_class] += 1  # Update confusion matrix

    # Compute weighted accuracy
    weighted_acc = np.trace(weights * confusion) / np.sum(weights * confusion)

    return weighted_acc


# ✅ Main evaluation function (Modified for "Diamond", "Plateau", "Decrescendo")
def evaluate_model(label_folder, output_folder):
    print("🔍 Evaluating model...")

    # Load label & output files
    label_files, output_files = find_challenge_files(label_folder, output_folder)
    shape_labels, valid_indices = load_shapes(label_files)  # Updated to load_shapes
    shape_binary_outputs, shape_scalar_outputs = load_classifier_outputs(output_files, valid_indices)

    # Find best threshold
    threshold = 0.5

    # Apply threshold
    shape_binary_outputs = (shape_scalar_outputs >= threshold).astype(int)

    # Compute evaluation metrics
    auroc_diamond, auprc_diamond, auroc_plateau, auprc_plateau, auroc_decrescendo, auprc_decrescendo = compute_auc(shape_labels, shape_scalar_outputs)
    shape_f_measure, shape_f_measure_classes = compute_f_measure(shape_labels, shape_binary_outputs)
    shape_accuracy, shape_accuracy_classes = compute_accuracy(shape_labels, shape_binary_outputs)
    shape_weighted_accuracy = compute_weighted_accuracy(shape_labels, shape_binary_outputs)

    return ["Diamond", "Plateau", "Decrescendo"], \
           [auroc_diamond, auroc_plateau, auroc_decrescendo], \
           [auprc_diamond, auprc_plateau, auprc_decrescendo], \
           shape_f_measure, shape_f_measure_classes, \
           shape_accuracy, shape_accuracy_classes, shape_weighted_accuracy

# ✅ Print & Save scores (Modified for "Diamond", "Plateau", "Decrescendo")
def print_and_save_scores(filename, shape_scores):
    classes, auroc, auprc, f_measure, f_measure_classes, accuracy, accuracy_classes, weighted_accuracy = shape_scores
    
    # Compute the overall AUROC and AUPRC as the mean of all classes
    total_auroc = np.mean(auroc)
    total_auprc = np.mean(auprc)

    output_string = f"""
#Shape scores
AUROC,AUPRC,F-measure,Accuracy,Weighted Accuracy
{total_auroc:.3f},{total_auprc:.3f},{f_measure:.3f},{accuracy:.3f},{weighted_accuracy:.3f}

#Shape scores (per class)
Classes,Diamond,Plateau,Decrescendo
AUROC,{auroc[0]:.3f},{auroc[1]:.3f},{auroc[2]:.3f}
AUPRC,{auprc[0]:.3f},{auprc[1]:.3f},{auprc[2]:.3f}
F-measure,{f_measure_classes[0]:.3f},{f_measure_classes[1]:.3f},{f_measure_classes[2]:.3f}
Accuracy,{accuracy_classes[0]:.3f},{accuracy_classes[1]:.3f},{accuracy_classes[2]:.3f}
"""

    # ✅ Print results to console
    print(output_string)

    # ✅ Save to file
    with open(filename, 'w') as f:
        f.write(output_string.strip())
    print(f"✅ Scores saved to {filename}")

# ✅ Run the evaluation script
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python evaluate_model.py <label_folder> <output_folder> <scores.csv>")
        sys.exit(1)

    shape_scores = evaluate_model(sys.argv[1], sys.argv[2])
    print_and_save_scores(sys.argv[3], shape_scores)

    print("✅ Model Evaluation Completed. Check scores.csv for detailed results.")

