import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib
matplotlib.use('Agg')
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from models.Generate_Model import GenerateModel
from collections import defaultdict


def fairness_test(loaded_model, test_loader, best_checkpoint_path):

    model = loaded_model
    # model = torch.nn.DataParallel(model).cuda()
    model.load_state_dict(torch.load(best_checkpoint_path))

    correct_predictions = 0
    total_samples = 0

    correct_predictions_group = defaultdict(lambda: defaultdict(int))
    total_samples_group = defaultdict(lambda: defaultdict(int))
    group_accuracy = defaultdict(lambda: defaultdict(int))

    true_positive_group = defaultdict(int)
    false_positive_group = defaultdict(int)
    false_negative_group = defaultdict(int)
    true_negative_group = defaultdict(int)

    model.eval()

    with torch.no_grad():
        for i, (image_tensor, label_target, label_sensitive, image_name) in enumerate(test_loader):

            images = image_tensor.cuda()
            label_target = label_target.cuda()
            label_sensitive = label_sensitive.cuda()

            output = torch.sigmoid(model(images))

            # Binary predictions (0 or 1)
            predictions = (output >= 0.5).float()

            # Update correct predictions and total samples for all samples
            correct_predictions += (predictions == label_target).sum().item()
            total_samples += label_target.size(0)

            # Update correct predictions and total samples for each group
            for j, sensitive in enumerate(label_sensitive):
                target_label = label_target[j].item()
                sensitive_label = sensitive.item()
                pred_label = predictions[j].item()
                
                correct_predictions_group[target_label][sensitive_label] += (pred_label == target_label)
                total_samples_group[target_label][sensitive_label] += 1

                if target_label == 1 and pred_label == 1:
                    true_positive_group[sensitive_label] += 1
                if target_label == 0 and pred_label == 1:
                    false_positive_group[sensitive_label] += 1
                if target_label == 1 and pred_label == 0:
                    false_negative_group[sensitive_label] += 1
                if target_label == 0 and pred_label == 0:
                    true_negative_group[sensitive_label] += 1

    # Calculate the overall accuracy
    accuracy = correct_predictions / total_samples
    
    # Calculate the accuracy, TPR, FPR for each group
    target = [0, 1]
    sensitive = [0, 1]
    tpr_group = {}
    fpr_group = {}
    all_group_accuracies = []

    for _target_label in target:
        for _sensitive_label in sensitive:
            group_samples = total_samples_group[_target_label][_sensitive_label]
            if group_samples > 0:
                group_accuracy[_target_label][_sensitive_label] = correct_predictions_group[_target_label][_sensitive_label] / group_samples
                all_group_accuracies.append(group_accuracy[_target_label][_sensitive_label])
                tpr_group[_sensitive_label] = true_positive_group[_sensitive_label] / (true_positive_group[_sensitive_label] + false_negative_group[_sensitive_label])
                fpr_group[_sensitive_label] = false_positive_group[_sensitive_label] / (false_positive_group[_sensitive_label] + true_negative_group[_sensitive_label])
            else:
                print(f'{"No samples for " + str(_target_label) + ", " + str(_sensitive_label):<25}')

    # Calculate  Equal Opportunity
    tpr_diff = abs(tpr_group[1] - tpr_group[0])
    fpr_diff = abs(fpr_group[1] - fpr_group[0])
    equalized_odds = (tpr_diff + fpr_diff) / 2.0

    return group_accuracy, accuracy, equalized_odds