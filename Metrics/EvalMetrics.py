import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from ..src.Model_Eval import evaluate

def aggregate_results(result_dir, output_file):
    """
    Aggregate results from test output files, evaluate overall metrics,
    and generate performance plots.

    Parameters:
    result_dir (str): Directory containing result `.pkl` files.
    output_file (str): File path to save aggregated results.

    Returns:
    None
    """
    file_list = [f for f in os.listdir(result_dir) if f.startswith("op_") and f.endswith(".pkl")]
    txt = ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19',
           '01', '20', '02', '03', '04', '05', '06', '07', '08', '09']

    all_ac = []  # Aggregated actual classes
    all_c = []  # Aggregated predictions
    all_scores = []  # Aggregated scores
    evals = []  # Evaluation metrics for each file

    for file_name in file_list:
        with open(os.path.join(result_dir, file_name), 'rb') as f:
            data = pickle.load(f)

        ac = data["AC"]
        c = data["C"]
        scores = data["scores"]
        eval_values = data["evalValues"]

        all_ac.append(ac)
        all_c.append(c)
        all_scores.append(scores)
        evals.append(eval_values)

    # Flatten aggregated lists
    all_ac = np.concatenate(all_ac)
    all_c = np.concatenate(all_c)
    all_scores = np.vstack(all_scores)

    # Evaluate overall parameters
    overall_eval = evaluate(all_ac, all_c)

    # Obtain performance curves
    fpr, tpr, _ = roc_curve(all_ac, all_scores[:, 1], pos_label=1)
    fpr0, tpr0, _ = roc_curve(all_ac, all_scores[:, 0], pos_label=0)
    auc_score = auc(fpr, tpr)

    # Plot ROC Curve
    plt.figure()
    plt.plot(fpr, tpr, label=f"Class 1 ROC (AUC = {auc_score:.2f})")
    plt.plot(fpr0, tpr0, label="Class 0 ROC")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Plot")
    plt.legend(loc="best")
    plt.show()

    # Plot F1 Scores for each file
    f1_scores = [e["F1_Score"] for e in evals]
    plt.figure()
    plt.scatter(range(1, len(file_list) + 1), f1_scores)
    for i, txt_label in enumerate(txt):
        plt.text(i + 1, f1_scores[i], txt_label)
    plt.xlabel("Image ID")
    plt.ylabel("F1 Score")
    plt.title("F1 Scores")
    plt.show()

    # Save aggregated results
    with open(output_file, 'wb') as f:
        pickle.dump({
            "AllAC": all_ac,
            "AllC": all_c,
            "AllScores": all_scores,
            "EvalValues": evals,
            "OverallEval": overall_eval,
            "ROC_FPR": fpr,
            "ROC_TPR": tpr,
            "ROC_FPR0": fpr0,
            "ROC_TPR0": tpr0
        }, f)
