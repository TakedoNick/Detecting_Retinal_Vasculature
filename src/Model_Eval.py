import numpy as np

def evaluate(actual, predicted):
    """
    Evaluate the performance of a classification model.

    Parameters:
    actual (numpy.ndarray): Column array with actual class labels (0 or 1).
    predicted (numpy.ndarray): Column array with predicted class labels (0 or 1).

    Returns:
    dict: Dictionary with performance metrics including accuracy, sensitivity,
          specificity, precision, recall, F1 score, and G-mean.
    """

    actual = np.array(actual).flatten()
    predicted = np.array(predicted).flatten()

    # Positive and negative indices
    idx = (actual == 1)

    # True positive, true negative, false positive, false negative
    tp = np.sum(actual[idx] == predicted[idx])
    tn = np.sum(actual[~idx] == predicted[~idx])
    fp = np.sum(actual[~idx] != predicted[~idx])
    fn = np.sum(actual[idx] != predicted[idx])

    # Positive and negative counts
    p = np.sum(idx)
    n = len(actual) - p
    N = p + n

    tp_rate = tp / p if p > 0 else 0
    tn_rate = tn / n if n > 0 else 0
    fp_rate = fp / n if n > 0 else 0

    # Metrics
    accuracy = (tp + tn) / N if N > 0 else 0
    sensitivity = tp_rate
    specificity = tn_rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = sensitivity
    f_measure = 2 * ((precision * recall) / (precision + recall)) if (precision + recall) > 0 else 0
    gmean = np.sqrt(tp_rate * tn_rate) if tp_rate > 0 and tn_rate > 0 else 0

    # Store metrics in a dictionary
    eval_metrics = {
        "True_Positive": tp,
        "False_Positive": fp,
        "False_Negative": fn,
        "True_Negative": tn,
        "True_Positive_Rate": tp_rate,
        "False_Positive_Rate": fp_rate,
        "Precision": precision,
        "Recall": recall,
        "F1_Score": f_measure,
        "Accuracy": accuracy,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "GMean": gmean
    }

    return eval_metrics