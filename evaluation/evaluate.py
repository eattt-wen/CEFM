import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    cohen_kappa_score
)

def evaluate_classifier(model, test_loader, device, model_name, output_dir):
    model.eval()
    total = 0
    correct = 0
    all_labels = []
    all_probs = []
    all_preds = []
    with torch.no_grad():
        for images, clinical_features, labels in test_loader:
            if model_name == 'image_classifier':
                inputs = images.to(device)
            else:
                inputs = clinical_features.to(device)
            labels = labels.to(device)
            outputs, _ = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)
    accuracy = 100.0 * correct / total
    cm = confusion_matrix(all_labels, all_preds)
    if cm.shape == (2, 2):
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
    else:
        tn, fp, fn, tp = 0, 0, 0, 0
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    sensitivity = recall
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    roc_auc = auc(fpr, tpr)
    report = classification_report(all_labels, all_preds)
    print(f'{model_name.capitalize()} Evaluation Results:')
    print(f'Accuracy: {accuracy:.2f}%')
    print(f'Balanced Accuracy: {balanced_acc:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Kappa Coefficient: {kappa:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall (Sensitivity): {recall:.4f}')
    print(f'Specificity: {specificity:.4f}')
    print(f'AUC: {roc_auc:.4f}')
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name.replace("_", " ").title()} ROC Curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(output_dir, f'{model_name}_roc_curve.png'))
    plt.close()
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name.replace("_", " ").title()} Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))
    plt.close()
    results_file = os.path.join(output_dir, f'{model_name}_evaluation.txt')
    with open(results_file, 'w') as f:
        f.write(f'{model_name.upper()} Evaluation Results\n')
        f.write('=' * 50 + '\n\n')
        f.write(f'Accuracy: {accuracy:.2f}%\n')
        f.write(f'Balanced Accuracy: {balanced_acc:.4f}\n')
        f.write(f'F1 Score: {f1:.4f}\n')
        f.write(f'Kappa Coefficient: {kappa:.4f}\n')
        f.write(f'Precision: {precision:.4f}\n')
        f.write(f'Recall (Sensitivity): {recall:.4f}\n')
        f.write(f'Specificity: {specificity:.4f}\n')
        f.write(f'AUC: {roc_auc:.4f}\n\n')
        f.write(f'Confusion Matrix:\n{cm}\n\n')
        f.write(f'Classification Report:\n{report}\n')
    metrics_df = pd.DataFrame({
        'Metric': [
            'Accuracy (%)', 'Balanced Accuracy', 'F1 Score', 'Kappa Coefficient',
            'Precision', 'Recall (Sensitivity)', 'Specificity', 'AUC'
        ],
        'Value': [
            accuracy, balanced_acc, f1, kappa, precision, recall, specificity, roc_auc
        ]
    })
    metrics_df.to_csv(os.path.join(output_dir, f'{model_name}_metrics.csv'), index=False)
    return {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1': f1,
        'kappa': kappa,
        'precision': precision,
        'recall': recall,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'auc': roc_auc,
        'cm': cm,
        'report': report,
        'fpr': fpr,
        'tpr': tpr,
        'all_labels': all_labels,
        'all_probs': all_probs,
        'all_preds': all_preds
    }
