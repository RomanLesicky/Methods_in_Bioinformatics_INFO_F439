"""
Overall 4 changes were made to fourth script of the code. 

  1. AUROC and AUPRC are now computed from P(positive class), not from hard labels like there were before.
     Essentially the original code collapsed the softmax to argmax before feeding the scores to roc_auc_score / average_precision_score which makes those metrics degenerate.
     So this was patched so that AUROC and AUPRC can be also reported in the implemented version. These are actually interesting additions due to their nature as 
     threshold-free performance metrics which were totally unreported in the original publication. 

  2. There was a hardcoded `x = range(120)`, which was wrong for the human/ppi dataset. In this version now, the code uses sorted(history.keys()) to get the correct dimensions. 
     This way the code adapt to however many epochs were actually saved without having to hardcode values for each dataset.

  3. The biggest change that was brought concerns a "peculiarity" which was found only in the ppi/humain dataset. Wherein this particular dataset, idx_val == idx_test were equal
     in the loader branch reporting performances metric values of the best epoch by test accuracy and not validation accuracy. Which theoretically could have lead to a slight 
     over-inflation of results. The current code reports the original finds, the corrected findings and reports a delta for only this dataset. Since, this "peculiarity" was 
     present in this singular example. 

  4. Add a CLI argument for ease of use. So now running `python script_4_evaluation_plots.py` runs the script on all datasets while `python script_4_evaluation_plots.py --dataset ppi`
     will do it on only the specified one. 
"""

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
)

from code.ResultSaving import ResultSaving


DATASETS = ['ppi', 'human', 'e.coli', 'drosophila', 'c.elegan']
HIDDEN_LAYERS = 2
RESULT_FOLDER = './result/GraphBert/'


def to_numpy(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def evaluate_epoch(record):
    """Compute all metrics for a single saved epoch record"""
    y_true = to_numpy(record['test_acc_data']['true_y'])

    test_op = record['test_op']
    if not torch.is_tensor(test_op):
        test_op = torch.tensor(test_op)
    probs = F.softmax(test_op, dim=1).detach().cpu().numpy()

    # Hard predictions that are used for accuracy, confusion matrix, MCC, F1, etc...
    y_pred = probs.argmax(axis=1)

    # Continuous positive-class score which is the correct input for AUROC / AUPRC
    # This is the change number 1 
    y_score = probs[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )

    return {
        'accuracy':    accuracy_score(y_true, y_pred),
        'sensitivity': tp / (tp + fn) if (tp + fn) else 0.0,
        'specificity': tn / (tn + fp) if (tn + fp) else 0.0,
        'precision':   precision,
        'recall':      recall,
        'f1':          f1,
        'mcc':         matthews_corrcoef(y_true, y_pred),
        'auroc':       roc_auc_score(y_true, y_score),
        'auprc':       average_precision_score(y_true, y_score),
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
    }


def run_dataset(dataset_name):
    result_obj = ResultSaving('', '')
    result_obj.result_destination_folder_path = RESULT_FOLDER
    result_obj.result_destination_file_name = f'{dataset_name}_{HIDDEN_LAYERS}'

    try:
        history = result_obj.load()
    except (FileNotFoundError, IOError):
        print(f'\n[{dataset_name}] no saved history at '
              f'{RESULT_FOLDER}{dataset_name}_{HIDDEN_LAYERS} — skipping')
        return

    # Change number 2 
    epochs = sorted(history.keys())
    if not epochs:
        print(f'\n[{dataset_name}] history is empty — skipping')
        return

    # Change number 3 

    # "Legacy" selector for lack of a better name, which selects the best epoch by test accuracy (what the original script did)
    best_test_epoch = max(epochs, key=lambda e: history[e]['acc_test'])
    # Corrected selector which selects the best epoch by validation accuracy
    best_val_epoch  = max(epochs, key=lambda e: history[e]['acc_val'])

    legacy    = evaluate_epoch(history[best_test_epoch])
    corrected = evaluate_epoch(history[best_val_epoch])

    # Detect whether the comparison is mechanically meaningful
    # On datasets where idx_val == idx_test, acc_val == acc_test at every epoch so argmax picks the same epoch and the two columns are identical
    same_epoch = (best_test_epoch == best_val_epoch)
    identical  = same_epoch and all(
        np.isclose(legacy[k], corrected[k])
        for k in legacy if isinstance(legacy[k], float)
    )

    print(f'\n{"=" * 72}')
    print(f'DATASET: {dataset_name}   (epochs in history: {len(epochs)}, '
          f'range {epochs[0]}..{epochs[-1]})')
    print(f'  legacy selector    (argmax acc_test) -> epoch {best_test_epoch}')
    print(f'  corrected selector (argmax acc_val)  -> epoch {best_val_epoch}')
    if identical:
        print('  NOTE: legacy and corrected are identical on this dataset.')
        print('        This is expected when the active loader branch sets')
        print('        idx_val == idx_test. The comparison is only')
        print('        informative on ppi in the current DatasetLoader.py.')
    print(f'{"-" * 72}')
    print(f'  {"metric":<12s} {"legacy":>12s} {"corrected":>12s} {"delta":>12s}')
    for k in ['accuracy', 'sensitivity', 'specificity', 'precision',
              'recall', 'f1', 'mcc', 'auroc', 'auprc']:
        delta = corrected[k] - legacy[k]
        print(f'  {k:<12s} {legacy[k]:>12.4f} {corrected[k]:>12.4f} '
              f'{delta:>+12.4f}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--dataset',
        default='all',
        help=f'Dataset name, or "all" to loop over {DATASETS}. Default: all.',
    )
    args = parser.parse_args()

    if args.dataset == 'all':
        for d in DATASETS:
            run_dataset(d)
    else:
        run_dataset(args.dataset)


if __name__ == '__main__':
    main()