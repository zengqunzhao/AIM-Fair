import argparse
import datetime
import shutil
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dataloader.load_data_CelebA import *
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import random
from models.Generate_Model import GenerateModel
from models.Fairness_Metrics import fairness_test
from fractions import Fraction

parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=8)
parser.add_argument('--pre-train-epochs', type=int, default=15)
parser.add_argument('--pre-train-lr', type=float, default=0.01)
parser.add_argument('--fine-tune-epochs', type=int, default=10)
parser.add_argument('--fine-tune-lr', type=float, default=0.05)
parser.add_argument('--batch-size', type=int, default=48)
parser.add_argument('--weight-decay', type=float, default=1e-4)
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--milestones', nargs='+', type=int)
parser.add_argument('--seeds', nargs='+', type=int)
parser.add_argument('--job-id', type=str)
parser.add_argument('--target-attribute', type=str, default="Smiling")
parser.add_argument('--sensitive-attribute', type=str, default="Male")
parser.add_argument('--bias-degree', type=str, default="1/9")
parser.add_argument('--image-size', type=int, default=224)
parser.add_argument('--real-data-path', type=str)
parser.add_argument('--synthetic-data-root', type=str)
parser.add_argument('--number-train-data-real', type=int, default=20000)
parser.add_argument('--number-balanced-synthetic-0-0', type=int, default=5000)
parser.add_argument('--number-balanced-synthetic-0-1', type=int, default=5000)
parser.add_argument('--number-balanced-synthetic-1-0', type=int, default=5000)
parser.add_argument('--number-balanced-synthetic-1-1', type=int, default=5000)
parser.add_argument('--number-imbalanced-synthetic-0-0', type=int, default=1000)
parser.add_argument('--number-imbalanced-synthetic-0-1', type=int, default=9000)
parser.add_argument('--number-imbalanced-synthetic-1-0', type=int, default=9000)
parser.add_argument('--number-imbalanced-synthetic-1-1', type=int, default=1000)
parser.add_argument('--top-k', type=int, default=40)

args = parser.parse_args()

print('##### Training Setting #####')
for k, v in vars(args).items():
    print(k,'=',v)
print('############################')

now = datetime.datetime.now()
print("Start Job Time: ", now.strftime("%y-%m-%d %H:%M"))

def main(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    log_txt_path_pretrain = f"./log/{args.job_id}_{str(seed)}_log_pretrain.txt"
    log_curve_path_pretrain = f"./log/{args.job_id}_{str(seed)}_log_pretrain.png"
    log_txt_path_finetune = f"./log/{args.job_id}_{str(seed)}_log_finetune.txt"
    log_curve_path_finetune = f"./log/{args.job_id}_{str(seed)}_log_finetune.png"
    checkpoint_path_pretrain = f"./checkpoint/pretrain_{args.target_attribute}_{args.sensitive_attribute}_{str(seed)}.pth"
    checkpoint_path_finetune = f"./checkpoint/finetune_{args.target_attribute}_{args.sensitive_attribute}_{str(seed)}.pth"
    best_checkpoint_path_pretrain= f"./checkpoint/pretrain_best_{args.target_attribute}_{args.sensitive_attribute}_{str(seed)}.pth"
    best_checkpoint_path_finetune = f"./checkpoint/finetune_best_{args.target_attribute}_{args.sensitive_attribute}_{str(seed)}.pth"
    recorder_pretrain = RecorderMeter(args.pre_train_epochs)
    recorder_finetune = RecorderMeter(args.fine_tune_epochs)
    gradient_difference_descending_1 = False   # Select the parames with lower difference
    gradient_difference_descending_2 = True    # Select the parames with higher difference

    train_real_imbalanced_loader, train_synthetic_balanced_loader, train_synthetic_imbalanced_loader, val_loader, test_loader = _data_loader(args=args)

    # print params   
    with open(log_txt_path_pretrain, 'a') as f:
        f.write('########## Training Settings ########## \n')
        for k, v in vars(args).items():
            f.write(str(k) + '=' + str(v) + '\n')
        f.write('####################################### \n')
    print("##### Training Seed:", str(seed), "#####")

    ########################################
    ############# Pre-Training #############
    ########################################
    model = GenerateModel()
    model = torch.nn.DataParallel(model).cuda()

    criterion = nn.BCEWithLogitsLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.pre_train_lr,  momentum=0.9,  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    best_acc = 0  
    for epoch in range(0, args.pre_train_epochs):

        inf = '********************' + str(epoch) + '********************'
        current_learning_rate_0 = optimizer.state_dict()['param_groups'][0]['lr']

        with open(log_txt_path_pretrain, 'a') as f:
            f.write(inf + '\n')
            print(inf)
            f.write('Current learning rate: ' + str(current_learning_rate_0) + '\n')
            print('Current learning rate: ', current_learning_rate_0)         
            
        train_acc, train_los = train(train_real_imbalanced_loader, model, criterion, optimizer, epoch, args, log_txt_path_pretrain)
        val_acc, val_los = validate(val_loader, model, criterion, args, log_txt_path_pretrain)
        scheduler.step()

        # select the best model based on the performance on Val Set
        is_best = val_acc >= best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint(model.state_dict(), is_best, checkpoint_path_pretrain, best_checkpoint_path_pretrain)

        # print and save log
        recorder_pretrain.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder_pretrain.plot_curve(log_curve_path_pretrain)
        print('The best accuracy: {:.3f}'.format(best_acc))
        with open(log_txt_path_pretrain, 'a') as f:
            f.write('The best accuracy: ' + str(best_acc) + '\n')

    group_accuracy, accuracy, equalized_odds = fairness_test(model, test_loader, best_checkpoint_path_pretrain)
    Acc_0_0, Acc_0_1, Acc_1_0, Acc_1_1, Acc_overall, E_Odd = 0, 0, 0, 0, 0, 0
    Acc_0_0 += group_accuracy[0][0]
    Acc_0_1 += group_accuracy[0][1]
    Acc_1_0 += group_accuracy[1][0]
    Acc_1_1 += group_accuracy[1][1]
    Acc_overall += accuracy
    E_Odd += equalized_odds
    print("*" * 23)
    print("Results on Biased Real Data")
    print(f"Seed: {str(seed)}")
    print(f'{"Accuracy for (0, 0):":<25} {group_accuracy[0][0]*100:>6.2f}')
    print(f'{"Accuracy for (0, 1):":<25} {group_accuracy[0][1]*100:>6.2f}')
    print(f'{"Accuracy for (1, 0):":<25} {group_accuracy[1][0]*100:>6.2f}')
    print(f'{"Accuracy for (1, 1):":<25} {group_accuracy[1][1]*100:>6.2f}')
    print(f'{"Overall Accuracy:":<25} {accuracy*100:>6.2f}')
    print(f'{"Worst-group Accuracy:":<25} {min(group_accuracy[0][0], group_accuracy[0][1], group_accuracy[1][0], group_accuracy[1][1])*100:>6.2f}')
    print(f'{"Equalized Odds:":<25} {equalized_odds*100:>6.2f}')
    print(f'{"STD of Group Accuracies:":<25} {np.std([group_accuracy[0][0], group_accuracy[0][1], group_accuracy[1][0], group_accuracy[1][1]])*100:>6.2f}')
    print("*" * 23)

    ########################################
    ######### AIM-Fair Fine-Tuning #########
    ########################################
    model.load_state_dict(torch.load(best_checkpoint_path_pretrain))
    
    gradients_real_biased = compute_gradients(train_real_imbalanced_loader, criterion, best_checkpoint_path_pretrain)
    gradients_synthetic_biased = compute_gradients(train_synthetic_imbalanced_loader, criterion, best_checkpoint_path_pretrain)
    gradients_synthetic_fair = compute_gradients(train_synthetic_balanced_loader, criterion, best_checkpoint_path_pretrain)

    gradient_difference_1 = compute_mean_difference(gradients_real_biased, gradients_synthetic_biased)
    gradient_difference_2 = compute_mean_difference(gradients_synthetic_biased, gradients_synthetic_fair)
    common_keys = top_k_common_elements(gradient_difference_1, gradient_difference_descending_1, gradient_difference_2, gradient_difference_descending_2, args.top_k)

    for name, param in model.named_parameters():
        param.requires_grad = False
    for name, param in model.named_parameters():
        if name in common_keys:
            param.requires_grad = True

    optimizer = torch.optim.SGD(model.parameters(), lr=args.fine_tune_lr, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
   
    best_acc = 0   
    for epoch in range(0, int(args.fine_tune_epochs)):
        inf = '********************' + str(epoch) + '********************'
        current_learning_rate_0 = optimizer.state_dict()['param_groups'][0]['lr']

        with open(log_txt_path_finetune, 'a') as f:
            f.write(inf + '\n')
            print(inf)
            f.write('Current learning rate: ' + str(current_learning_rate_0) + '\n')
            print('Current learning rate: ', current_learning_rate_0)         
    
        train_acc, train_los = train(train_synthetic_balanced_loader, model, criterion, optimizer, epoch, args, log_txt_path_finetune)
        val_acc, val_los = validate(val_loader, model, criterion, args, log_txt_path_finetune)
        scheduler.step()

        # select the best model based on the performance on Val Set
        is_best = val_acc > best_acc
        best_acc = max(val_acc, best_acc)
        save_checkpoint(model.state_dict(), is_best, checkpoint_path_finetune, best_checkpoint_path_finetune)

        # print and save log
        recorder_finetune.update(epoch, train_los, train_acc, val_los, val_acc)
        recorder_finetune.plot_curve(log_curve_path_finetune)
        print('The best accuracy: {:.3f}'.format(best_acc))
        with open(log_txt_path_finetune, 'a') as f:
            f.write('The best accuracy: ' + str(best_acc) + '\n')

    group_accuracy, accuracy, equalized_odds = fairness_test(model, test_loader, best_checkpoint_path_finetune)

    return group_accuracy, accuracy, equalized_odds


def _data_loader(args):

    bias_degree =  float(Fraction(args.bias_degree))
    project_path = "./annotations/"

    if args.number_train_data_real != 0:
        train_real_annotation_path = f"{project_path}CelebA_Train_{args.target_attribute}_{args.sensitive_attribute}_{args.number_train_data_real}_{(bias_degree / (bias_degree + 1) / 2):.2f}.csv"
    else:
        train_real_annotation_path = None
    val_annotation_path = f"{project_path}CelebA_Val.csv"
    test_annotation_path = f"{project_path}CelebA_Test.csv"
    synthetic_data_path = f"{args.synthetic_data_root}{args.target_attribute}_{args.sensitive_attribute}/"
    
    real_imbalanced = Dataloader_CelebA_RealAndSynthetic(
                            dataset_path_real=args.real_data_path,
                            annotation_path_real=train_real_annotation_path,
                            dataset_path_synthetic=synthetic_data_path,
                            target=args.target_attribute, 
                            sensitive=args.sensitive_attribute, 
                            img_size=args.image_size,
                            real_data_number_train=args.number_train_data_real,
                            synthetic_number_train_0_0=0,
                            synthetic_number_train_0_1=0,
                            synthetic_number_train_1_0=0,
                            synthetic_number_train_1_1=0)
    synthetic_balanced = Dataloader_CelebA_RealAndSynthetic(
                            dataset_path_real=args.real_data_path,
                            annotation_path_real=train_real_annotation_path,
                            dataset_path_synthetic=synthetic_data_path,
                            target=args.target_attribute, 
                            sensitive=args.sensitive_attribute, 
                            img_size=args.image_size,
                            real_data_number_train=0,
                            synthetic_number_train_0_0=args.number_balanced_synthetic_0_0,
                            synthetic_number_train_0_1=args.number_balanced_synthetic_0_1,
                            synthetic_number_train_1_0=args.number_balanced_synthetic_1_0,
                            synthetic_number_train_1_1=args.number_balanced_synthetic_1_1)
    synthetic_imbalanced = Dataloader_CelebA_RealAndSynthetic(
                            dataset_path_real=args.real_data_path,
                            annotation_path_real=train_real_annotation_path,
                            dataset_path_synthetic=synthetic_data_path,
                            target=args.target_attribute, 
                            sensitive=args.sensitive_attribute, 
                            img_size=args.image_size,
                            real_data_number_train=0,
                            synthetic_number_train_0_0=args.number_imbalanced_synthetic_0_0,
                            synthetic_number_train_0_1=args.number_imbalanced_synthetic_0_1,
                            synthetic_number_train_1_0=args.number_imbalanced_synthetic_1_0,
                            synthetic_number_train_1_1=args.number_imbalanced_synthetic_1_1)                
    val_data = Dataloader_CelebA_Real(
                            mode="val",
                            dataset_path=args.real_data_path,
                            annotation_path=val_annotation_path,
                            target=args.target_attribute, 
                            sensitive=args.sensitive_attribute, 
                            img_size=args.image_size)
    test_data = Dataloader_CelebA_Real(
                            mode="test",
                            dataset_path=args.real_data_path,
                            annotation_path=test_annotation_path,
                            target=args.target_attribute, 
                            sensitive=args.sensitive_attribute, 
                            img_size=args.image_size)

    real_imbalanced_loader = torch.utils.data.DataLoader(real_imbalanced,
                                                         batch_size=args.batch_size,
                                                         shuffle=True,
                                                         num_workers=args.workers,
                                                         pin_memory=True,
                                                         drop_last=True)
    synthetic_balanced_loader = torch.utils.data.DataLoader(synthetic_balanced,
                                                            batch_size=args.batch_size,
                                                            shuffle=True,
                                                            num_workers=args.workers,
                                                            pin_memory=True,
                                                            drop_last=True)
    synthetic_imbalanced_loader = torch.utils.data.DataLoader(synthetic_imbalanced,
                                                              batch_size=args.batch_size,
                                                              shuffle=True,
                                                              num_workers=args.workers,
                                                              pin_memory=True,
                                                              drop_last=True)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data,
                                              batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=True)

    return real_imbalanced_loader, synthetic_balanced_loader, synthetic_imbalanced_loader, val_loader, test_loader


def train(train_loader, model, criterion, optimizer, epoch, args, log_txt_path):
    losses = AverageMeter('Loss', ':.4f')
    accuracy = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(train_loader), [losses, accuracy], prefix="Epoch: [{}]".format(epoch), log_txt_path=log_txt_path)
    model.train()

    for i, (image_tensor, label_target, label_sensitive, image_name) in enumerate(train_loader):

        images = image_tensor.cuda() 
        target = label_target.cuda().float()

        output = model(images)

        loss = criterion(output, target)
        
        # measure accuracy and record loss
        accuracy_batch = binary_accuracy(output, target)
        losses.update(loss.item(), images.size(0))
        accuracy.update(accuracy_batch, images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print loss and accuracy
        if i % args.print_freq == 0:
            progress.display(i)

    return accuracy.avg, losses.avg


def validate(val_loader, model, criterion, args, log_txt_path):
    losses = AverageMeter('Loss', ':.4f')
    accuracy = AverageMeter('Accuracy', ':6.3f')
    progress = ProgressMeter(len(val_loader), [losses, accuracy], prefix='Val: ', log_txt_path=log_txt_path)
    model.eval()

    with torch.no_grad():
        for i, (image_tensor, label_target, label_sensitive, image_name) in enumerate(val_loader):

            images = image_tensor.cuda() 
            target = label_target.cuda().float()

            output = model(images)

            loss = criterion(output, target)

            # measure accuracy and record loss
            accuracy_batch = binary_accuracy(output, target)
            losses.update(loss.item(), images.size(0))
            accuracy.update(accuracy_batch, images.size(0))

            if i % args.print_freq == 0:
                progress.display(i)

        print('Current Accuracy: {accuracy.avg:.3f}'.format(accuracy=accuracy))
        with open(log_txt_path, 'a') as f:
            f.write('Current Accuracy: {accuracy.avg:.3f}'.format(accuracy=accuracy) + '\n')

    return accuracy.avg, losses.avg


def compute_gradients(data_loader, criterion, checkpoint_path):

    model = GenerateModel()
    model = torch.nn.DataParallel(model).cuda() 
    model.load_state_dict(torch.load(checkpoint_path))

    aggregated_grads = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
    for param in model.parameters():
        param.requires_grad = True
    model.train()
    for inputs, labels, label_sensitive, image_name in data_loader:
        inputs = inputs.cuda()
        labels = labels.cuda().float()
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                aggregated_grads[name] += param.grad.detach()
    num_batches = len(data_loader)
    for name in aggregated_grads:
        aggregated_grads[name] /= num_batches
    return aggregated_grads


def compute_mean_difference(grad1, grad2):
    mean_abs_diffs = {}
    for name in grad1.keys():
        # Flatten the gradients and convert them to numpy arrays
        g1 = grad1[name].flatten().cpu().numpy()
        g2 = grad2[name].flatten().cpu().numpy()
        # Calculate the absolute difference and mean
        mean_abs_diff = np.mean(np.abs(g1 - g2))
        mean_abs_diffs[name] = mean_abs_diff
    return mean_abs_diffs


def top_k_common_elements(dict1, reverse1, dict2, reverse2, k):
    # Sort each dictionary by values in descending order
    sorted_keys_1 = sorted(dict1, key=dict1.get, reverse=reverse1)[:k]
    sorted_keys_2 = sorted(dict2, key=dict2.get, reverse=reverse2)[:k]
    # Find the common elements in the top-k keys
    common_elements = set(sorted_keys_1).intersection(sorted_keys_2)
    return common_elements


def save_checkpoint(state, is_best, checkpoint_path, best_checkpoint_path):
    torch.save(state, checkpoint_path)
    if is_best:
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", log_txt_path=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.log_txt_path = log_txt_path

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print_txt = '\t'.join(entries)
        print(print_txt)
        with open(self.log_txt_path, 'a') as f:
            f.write(print_txt + '\n')

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def binary_accuracy(output, target):
    """Computes the accuracy for binary classification"""
    with torch.no_grad():
        probabilities = torch.sigmoid(output)
        pred = (probabilities >= 0.5).float()
        correct = pred.eq(target).sum().item()
        accuracy = correct / target.size(0) * 100.0
        return accuracy


class RecorderMeter(object):
    """Computes and stores the minimum loss value and its epoch index"""
    def __init__(self, total_epoch):
        self.reset(total_epoch)

    def reset(self, total_epoch):
        self.total_epoch = total_epoch
        self.current_epoch = 0
        self.epoch_losses = np.zeros((self.total_epoch, 2), dtype=np.float32)    # [epoch, train/val]
        self.epoch_accuracy = np.zeros((self.total_epoch, 2), dtype=np.float32)  # [epoch, train/val]

    def update(self, idx, train_loss, train_acc, val_loss, val_acc):
        self.epoch_losses[idx, 0] = train_loss * 50
        self.epoch_losses[idx, 1] = val_loss * 50
        self.epoch_accuracy[idx, 0] = train_acc
        self.epoch_accuracy[idx, 1] = val_acc
        self.current_epoch = idx + 1

    def plot_curve(self, save_path):

        title = 'the accuracy/loss curve of train/val'
        dpi = 80
        width, height = 1600, 800
        legend_fontsize = 10
        figsize = width / float(dpi), height / float(dpi)

        fig = plt.figure(figsize=figsize)
        x_axis = np.array([i for i in range(self.total_epoch)])  # epochs
        y_axis = np.zeros(self.total_epoch)

        plt.xlim(0, self.total_epoch)
        plt.ylim(0, 100)
        interval_y = 5
        interval_x = 1
        plt.xticks(np.arange(0, self.total_epoch + interval_x, interval_x))
        plt.yticks(np.arange(0, 100 + interval_y, interval_y))
        plt.grid()
        plt.title(title, fontsize=20)
        plt.xlabel('the training epoch', fontsize=16)
        plt.ylabel('accuracy', fontsize=16)

        y_axis[:] = self.epoch_accuracy[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle='-', label='train-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_accuracy[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle='-', label='valid-accuracy', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 0]
        plt.plot(x_axis, y_axis, color='g', linestyle=':', label='train-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        y_axis[:] = self.epoch_losses[:, 1]
        plt.plot(x_axis, y_axis, color='y', linestyle=':', label='valid-loss-x50', lw=2)
        plt.legend(loc=4, fontsize=legend_fontsize)

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        plt.close(fig)


if __name__ == '__main__':

    Acc_0_0, Acc_0_1, Acc_1_0, Acc_1_1, Acc_overall, E_Odd = 0, 0, 0, 0, 0, 0

    for seed in args.seeds:

        group_accuracy, accuracy, equalized_odds = main(seed)
        Acc_0_0 += group_accuracy[0][0]
        Acc_0_1 += group_accuracy[0][1]
        Acc_1_0 += group_accuracy[1][0]
        Acc_1_1 += group_accuracy[1][1]
        Acc_overall += accuracy
        E_Odd += equalized_odds

        print("*" * 23)
        print(f"Seed: {str(seed)}")
        print("Results After AIM-Fair Fine-Tuning")
        print(f'{"Accuracy for (0, 0):":<25} {group_accuracy[0][0]*100:>6.2f}')
        print(f'{"Accuracy for (0, 1):":<25} {group_accuracy[0][1]*100:>6.2f}')
        print(f'{"Accuracy for (1, 0):":<25} {group_accuracy[1][0]*100:>6.2f}')
        print(f'{"Accuracy for (1, 1):":<25} {group_accuracy[1][1]*100:>6.2f}')
        print(f'{"Overall Accuracy:":<25} {accuracy*100:>6.2f}')
        print(f'{"Worst-group Accuracy:":<25} {min(group_accuracy[0][0], group_accuracy[0][1], group_accuracy[1][0], group_accuracy[1][1])*100:>6.2f}')
        print(f'{"Equalized Odds:":<25} {equalized_odds*100:>6.2f}')
        print(f'{"STD of Group Accuracies:":<25} {np.std([group_accuracy[0][0], group_accuracy[0][1], group_accuracy[1][0], group_accuracy[1][1]])*100:>6.2f}')
        print("*" * 23)

    Average_0_0 = Acc_0_0 / len(args.seeds) * 100
    Average_0_1 = Acc_0_1 / len(args.seeds) * 100
    Average_1_0 = Acc_1_0 / len(args.seeds) * 100
    Average_1_1 = Acc_1_1 / len(args.seeds) * 100
    Average_Acc = Acc_overall / len(args.seeds) * 100
    W_G_Acc = min(Average_0_0, Average_0_1, Average_1_0, Average_1_1)
    E_Odd = E_Odd / len(args.seeds) * 100
    G_Acc_STD = np.std([Average_0_0, Average_0_1, Average_1_0, Average_1_1])
    
    print("#" * 35)
    print(f"Target Attribute: {args.target_attribute}; Sensitive Attribute: {args.sensitive_attribute}")
    print(f'{"Accuracy for (0, 0):":<25} {Average_0_0:>6.2f}')
    print(f'{"Accuracy for (0, 1):":<25} {Average_0_1:>6.2f}')
    print(f'{"Accuracy for (1, 0):":<25} {Average_1_0:>6.2f}')
    print(f'{"Accuracy for (1, 1):":<25} {Average_1_1:>6.2f}')
    print(f'{"Overall Accuracy:":<25} {Average_Acc:>6.2f}')
    print(f'{"Worst-group Accuracy:":<25} {W_G_Acc:>6.2f}')
    print(f'{"Equalized Odds:":<25} {E_Odd:>6.2f}')
    print(f'{"STD of Group Accuracies:":<25} {G_Acc_STD:>6.2f}')
    print("#" * 35)

    now = datetime.datetime.now()
    print("Finish Job Time: ", now.strftime("%y-%m-%d %H:%M"))


