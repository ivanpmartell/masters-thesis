import os
import sys
import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt


###########################################
# Command line interface
this_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
default_out = os.path.join(this_dir, "results_ours")
default_res = os.path.join(this_dir, "results.csv")

parser = argparse.ArgumentParser(description=r"This script will analyse a model's performance returning multiple metrics for binary classification")
parser.add_argument('--output',
        type = str,
        help = f'Folder where the analysis files will be saved. Default: Same folder as the results file.'
        'The output folder is where all the figures and analysis files will be saved on disk',
        default = default_out
        )
parser.add_argument('--results',
        type = str,
        help = f'Path for results file. Default: {default_res}. '
        'The results file is a csv created by one of the test scripts',
        default = default_res
        )
parser.add_argument('--threshold', 
        type=float, 
        help='Threshold for the percentage to be classified as promoter. Default: 0.5',
        default=0.5)
args = parser.parse_args()
if(args.output == ''):
    args.output = os.path.dirname(args.results)
###########################################
if not os.path.exists(args.output):
    os.makedirs(args.output)

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    cm = metrics.confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig(os.path.join(args.output, '%s.png' % title))
    return ax


def softmax(x, r=1):
    temp = np.exp(x)
    return (temp / np.sum(temp))[r]


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def threshold(x):
    if x < args.threshold:
        return 0
    else:
        return 1


def threshold_flipped(x):
    if x < args.threshold:
        return 1
    else:
        return 0


#TODO: Add the distribution curve analysis for optimal threshold
# For sigmoid now
def create_histogram(data, colors, n_bins, title, zoom=0):
    plt.figure()
    n, bins, patches = plt.hist(x=data,bins=n_bins, color=colors,
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title(title)
    extra = '-zoomed_%d' % zoom
    plt.ylim(top=np.max(n[zoom]))
    plt.savefig(os.path.join(args.output, "%s%s.png" % (title, extra)))

print("Reading results")
df = pd.read_csv(args.results)
if(len(df.columns) == 5):
    truths_pos = df[df.label == 1][['prediction_0', 'prediction_1']].to_numpy()
    scores_pos = np.apply_along_axis(softmax, 1, truths_pos, slice(None))
    truths_neg = df[df.label == 0][['prediction_0', 'prediction_1']].to_numpy()
    scores_neg = np.apply_along_axis(softmax, 1, truths_neg, slice(None))
    predictions = df[['prediction_0', 'prediction_1']].to_numpy()
    scores = np.concatenate([truths_pos[:, 1], truths_neg[:, 1]])
else:
    predictions = df['prediction'].to_numpy()
    scores = np.vectorize(sigmoid)(predictions)
labels = df['label'].to_numpy()
print("Finished reading results")

print("Creating histograms")
n_bins=50
for i in range(2):
    create_histogram(truths_pos, ['#0504aa','#7cfc00'], n_bins, 'Promoters scores', zoom=i)
    create_histogram(truths_neg, ['#0504aa','#7cfc00'], n_bins, 'Non-promoters scores', zoom=i)
    create_histogram((truths_pos[:,0], truths_pos[:,1], truths_neg[:,0], truths_neg[:,1]), ['darkblue','blue','green','lightgreen'], n_bins, 'Scores', zoom=i*2)
    create_histogram((truths_pos[:,0], truths_pos[:,1], truths_neg[:,0], truths_neg[:,1]), ['darkblue','blue','green','lightgreen'], n_bins, 'Scores', zoom=i*2+1)
    create_histogram((scores_pos[:,1], scores_neg[:,0]), ['#0504aa','#7cfc00'], n_bins, 'Truth scores promoters', zoom=i)
    create_histogram((truths_pos[:,1], truths_neg[:,0]), ['#0504aa','#7cfc00'], n_bins, 'Truth logits promoters', zoom=i)
    create_histogram((scores_pos[:,0], scores_neg[:,1]), ['#0504aa','#7cfc00'], n_bins, 'Truth scores non-promoters', zoom=i)
    create_histogram((truths_pos[:,0], truths_neg[:,1]), ['#0504aa','#7cfc00'], n_bins, 'Truth logits non-promoters', zoom=i)
create_histogram(predictions, ['#0504aa','#7cfc00'], n_bins, 'Prediction logits')
print("Histograms done")

print("Calculating metrics")
fpr, tpr, roc_thresholds = metrics.roc_curve(labels, scores)
roc_auc = metrics.auc(fpr, tpr)
precision, recall, pr_thresholds = metrics.precision_recall_curve(labels, scores)
avg_precision = metrics.average_precision_score(labels, scores)
print("Metrics calculated")

def f_beta_by_threshold(y_true, y_pred_pos, thres_nr=100):
    thresholds = [i / thres_nr for i in range(1, thres_nr, 1)]

    f_scores = []
    for thres in thresholds:
        y_pred_class = y_pred_pos > thres
        score = metrics.matthews_corrcoef(y_true, y_pred_class)
        f_scores.append(score)

    return thresholds, f_scores

betas = [0.1, 0.25, 0.5, 1.0, 1.25, 1.5, 2, 5, 10]

plt.figure()
fig, ax = plt.subplots()
thresholds, betascores = f_beta_by_threshold(labels, scores)
ax.plot(thresholds, betascores, label='β=1')
ax.set_title('mcc by threshold')
ax.set_xlabel('threshold')
ax.set_ylabel('mcc')
ax.legend(loc='right', bbox_to_anchor=(1.3,0.7))
plt.savefig(os.path.join(args.output, 'mcc_by_thresholds.png'))

def make_report(scores, threshold_func, title):
    thresholded_scores = np.vectorize(threshold_func)(scores)
    report = metrics.classification_report(labels, thresholded_scores, labels=[0, 1], target_names=['non-promoter', 'promoter'], digits=3)
    acc_score = metrics.accuracy_score(labels, thresholded_scores)
    balanced_acc_score = metrics.balanced_accuracy_score(labels, thresholded_scores)
    f1_score = metrics.f1_score(labels, thresholded_scores)
    mcc_score = metrics.matthews_corrcoef(labels, thresholded_scores)
    precision_score = metrics.precision_score(labels, thresholded_scores)
    recall_score = metrics.recall_score(labels, thresholded_scores)
    fbeta_score = metrics.fbeta_score(labels, thresholded_scores, beta=1.0)
    jaccard_score = metrics.jaccard_score(labels, thresholded_scores)

    txt_string = 'Metrics report for trained model (%s) .\nThreshold set to %.2f\n' % (args.results, args.threshold)+\
                report + '\nAccuracy score: %0.3f' % acc_score +\
                '\nBalanced Accuracy score: %0.3f' % balanced_acc_score +\
                '\nF1 score: %0.3f' % f1_score +\
                '\nMatthews Coefficient Correlation: %0.3f' % mcc_score +\
                '\nPrecision score: %0.3f' % precision_score +\
                '\nRecall score: %0.3f' % recall_score +\
                '\nF-beta score: %0.3f' % fbeta_score +\
                '\nJaccard Similarity score: %0.3f' % jaccard_score

    with open(os.path.join(args.output, '%s-report.txt' % title), 'w') as text_file:
        text_file.write(txt_string)
    
    np.set_printoptions(precision=3)
    # Plot non-normalized confusion matrix
    plot_confusion_matrix(labels, thresholded_scores, classes=['non-promoter', 'promoter'],
                          normalize=True, title='Confusion Matrix-%s' % title)

make_report(scores, threshold, 'softmax')
args.threshold = -2.9
make_report(predictions[:, 0], threshold_flipped, 'logits')

# PLOT AUC
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.3f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Area under the ROC curve')
plt.legend(loc="lower right")
plt.savefig(os.path.join(args.output, 'area_under_curve.png'))

# PLOT PRCURVE
plt.figure()
lw = 2
plt.plot(recall[:-1], precision[:-1], color='darkorange',
         lw=lw, label='PR curve (AP = %0.3f)' % avg_precision)
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall curve')
plt.legend(loc="upper right")
plt.savefig(os.path.join(args.output, 'precision_recall_curve.png'))

# PLOT PR VS THRESHOLDS
plt.figure()
plt.plot(pr_thresholds, precision[:-1], 'b--', label='Precision')
plt.plot(pr_thresholds, recall[:-1], 'g--', label='Recall')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Threshold')
plt.legend(loc='upper right')
plt.savefig(os.path.join(args.output, 'precision_recall_vs_thresholds.png'))