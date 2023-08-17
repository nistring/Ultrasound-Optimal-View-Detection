import os
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, precision_recall_fscore_support, accuracy_score
from matplotlib import pyplot as plt
import pickle
import xlsxwriter
from ..config import *

model_name = {
    # 'resnet18' : 'ResNet18 (w/ Aug.)',
    # 'resnet34' : 'ResNet34 (w/ Aug.)',
    # 'resnet50' : 'ResNet50 (w/ Aug.)',
    # 'resnet101' : 'ResNet101 (w/ Aug.)',
    # 'resnet152' : 'ResNet152 (w/ Aug.)',
    'resnet34_noAUG' : 'ResNet34',
    'resnet34' : 'ResNet34 (w/ Aug.)',
    'resnet34_GRU' : 'ResNet34 + GRU (w/ Aug.)',
    'resnet34_SCL' : 'ResNet34 + SCL (w/ Aug.)',
    'resnet34-u-resnet34' : 'ResNet34 + Unet(ResNet34) (w/ Aug.)',
}

def roc(result, model, plot : bool):
    """Draws ROC curve

    Args:
        result (dict): Contains ground truth and predicted classes.
        model (str): Name of model.
        plot (bool): Plot if indicated.

    Returns:
        _type_: _description_
    """
    true = result['true']
    pred = result['pred']
    fpr, tpr, _ = roc_curve(true, pred)
    AUROC = auc(fpr, tpr)

    if plot is True:
        plt.plot(fpr, tpr, label=f"{model_name[model]}, AUC={AUROC:.3f}")
        plt.plot([0, 1], [0, 1],'r--')

        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.grid(True)

    return AUROC

def prc(result, model, threshold, plot : bool):
    """Draws PR curve

    Args:
        result (dict): Contains ground truth and predicted classes.
        model (str): Name of model.
        threshold (_type_): Optimal threhold for calculating precision and recall.
        plot (bool): Plot if indicated.

    Returns:
        _type_: _description_
    """
    true = result['true']
    pred = result['pred']

    precision, recall, _ = precision_recall_curve(true, pred)

    pred_binary = np.ones_like(pred)
    pred_binary[pred < threshold] = 0
    p, r, f, _ = precision_recall_fscore_support(true, pred_binary, average='binary')
    acc = accuracy_score(true, pred_binary)

    AUPRC = auc(recall, precision)

    if plot is True:
        plt.plot(recall, precision, label=f"{model_name[model]}, AUPRC={AUPRC:.3f}")
        # plt.scatter(best_recall, best_precision, c='r')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.grid(True)
    
    return acc, p, r, f

def draw():
    start = 5

    data_list = os.listdir(IMG_EVAL_DIR)
    data_list.remove('val')
    workbook = xlsxwriter.Workbook(os.path.join(VIDEO_EVAL_DIR, 'results.xlsx'))
    precision = workbook.add_format({'num_format': '0.00000'})
    worksheet = workbook.add_worksheet()
    worksheet.write(1, 1, 'Threshold')
    worksheet.write(1, 2, 'FPS')
    worksheet.write(1, 3, 'Batch size')
    worksheet.write(1, 4, 'Best epoch')
    for i, data_type in enumerate(data_list):
        worksheet.write(0, 5*i+start, data_type)
        for j, metric in enumerate(['Accuracy', 'Precision', 'Recall', 'F1 score', 'AUROC']):
            worksheet.write(1, 5*i+start+j, metric)

    # Draw figures for selected models
    val_pred_dir = os.path.join(IMG_EVAL_DIR, 'val')
    threshold_dict = {}
    selected_model_list = model_name.keys()
    whole_model_list = list(model_name.keys())
    for model in os.listdir(val_pred_dir):
        model = os.path.splitext(model)[0]
        if model not in selected_model_list:
            whole_model_list.append(model)

    # Calculate optimal threshold of every model from validation dataset
    for i, model in enumerate(whole_model_list):
        with open(os.path.join(val_pred_dir, model+'.pickle'), 'rb') as f:
            prediction = pickle.load(f)
            true = prediction['true']
            pred = prediction['pred']

            precision, recall, thresholds = precision_recall_curve(true, pred)
            numerator = 2 * recall * precision
            denom = recall + precision

            f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
            max_f1_score = np.max(f1_scores)
            opt_idx = np.where(f1_scores==max_f1_score)[0][0]
            threshold = thresholds[opt_idx]
            threshold_dict[model] = threshold

            worksheet.write(i+2, 0, model)
            worksheet.write(i+2, 1, threshold)

        # Write inference fps
        fps = 0
        for data_type in data_list:
            with open(os.path.join(IMG_EVAL_DIR, data_type, model+'.pickle'), 'rb') as f:
                fps += pickle.load(f)['fps']
        worksheet.write(i+2, 2, fps/len(data_list))

    # test_cropped, GE, mindray
    for i, data_type in enumerate(data_list):
        
        pred_dir = os.path.join(IMG_EVAL_DIR, data_type)

        # ROC curve
        plt.figure()
        for j, model in enumerate(whole_model_list):
            with open(os.path.join(pred_dir, model+'.pickle'), 'rb') as f:
                AUROC = roc(pickle.load(f), model,
                            True if model in selected_model_list else False)
                worksheet.write(j+2, 5*i+start+4, AUROC)
        plt.title('ROC curve')
        plt.legend()
        plt.savefig(os.path.join(FIG_DIR, f'ROC_curve({data_type}).png'))

        # Precision Recall Curve
        plt.figure()
        for j, model in enumerate(whole_model_list):
            with open(os.path.join(pred_dir, model+'.pickle'), 'rb') as f:
                metrics = prc(pickle.load(f), model, threshold_dict[model],
                            True if model in selected_model_list else False)
                for k, metric in enumerate(metrics):
                    worksheet.write(j+2, 5*i+start+k, metric)
        plt.title('Precision recall curve')
        plt.legend(loc=3)
        plt.savefig(os.path.join(FIG_DIR, f'PRC({data_type}).png'))
    
    workbook.close()



if __name__ == '__main__':
    draw()