import os
import datetime
import time
import warnings
import numpy as np
import random
import torch
import torch.utils.data
import torchvision
import utils
from torch import nn
import torchvision.transforms as transforms
from tiny_imagenet_dataset import TinyImageNet
from imagenet_ipc import ImageFolderIPC
import torch.nn.functional as F
from tqdm import tqdm
import json

warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler")

def get_args_parser(add_help=True):
    import argparse

    parser = argparse.ArgumentParser(description="CCFS on Tiny ImageNet", add_help=add_help)

    parser.add_argument("--data-path", default=None, type=str, help="path to TinyImageNet data folder")
    parser.add_argument("--filter-model", default="resnet18", type=str, help="filter model name")
    parser.add_argument("--teacher-model", default="resnet18", type=str, help="teacher model name")
    parser.add_argument("--teacher-path", default=None, type=str, help="path to teacher model")
    parser.add_argument("--eval-model", default="resnet18", type=str, help="model for final evaluation")
    
    parser.add_argument("--device", default="cuda", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=64, type=int, help="Batch size")
    parser.add_argument("--epochs", default=90, type=int, metavar="N", help="# training epochs for both the filter and the evaluation model")
    parser.add_argument("-j", "--workers", default=4, type=int, metavar="N", help="number of data loading workers (default: 16)")
    parser.add_argument("--opt", default="sgd", type=str, help="optimizer")
    parser.add_argument("--lr", default=0.1, type=float, help="initial learning rate")
    parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
    parser.add_argument("--wd", "--weight-decay", default=1e-4, type=float, metavar="W", help="weight decay (default: 1e-4)", dest="weight_decay")
    parser.add_argument("--lr-scheduler", default="steplr", type=str, help="the lr scheduler (default: steplr)")
    parser.add_argument("--lr-warmup-epochs", default=0, type=int, help="the number of epochs to warmup (default: 0)")
    parser.add_argument("--lr-warmup-method", default="constant", type=str, help="the warmup method (default: constant)")
    parser.add_argument("--lr-warmup-decay", default=0.01, type=float, help="the decay for lr")
    parser.add_argument("--lr-step-size", default=30, type=int, help="decrease lr every step-size epochs")
    parser.add_argument("-T", "--temperature", default=20, type=float, help="temperature for distillation loss")
    parser.add_argument("--print-freq", default=1000, type=int, help="print frequency")
    # CCFS parameters
    parser.add_argument("--image-per-class", default=50, type=int, help="number of synthetic images per class")
    parser.add_argument("--distill-data-path", default=None, type=str, help="path to already distilled data")
    parser.add_argument('--alpha', type=float, default=0.2, help='Distillation portion')
    parser.add_argument('--curriculum-num', type=int, default=None, help='Number of curricula')
    parser.add_argument('--select-misclassified', action='store_true', help='Selection strategy in coarse stage')
    parser.add_argument('--select-method', type=str, default='simple', choices=['random', 'hard', 'simple'], help='Selection strategy in fine stage')
    parser.add_argument('--balance', action='store_true', help='Whether to balance the amount of the synthetic data between classes')
    parser.add_argument('--score', type=str, default='forgetting', choices=['logits', 'forgetting'], help='Difficulty score used in fine stage')
    parser.add_argument('--score-path', type=str, default=None, help='Path to the difficulty score')
    parser.add_argument("--output-dir", default=".", type=str, help="path to save outputs")
    parser.add_argument("--num-eval", default=1, type=int, help="number of evaluations")

    return parser

def load_data(args):
    # Data loading code
    normalize = transforms.Normalize(mean=[0.4802, 0.4481, 0.3975],
                                     std=[0.2302, 0.2265, 0.2262])
    print("Loading distilled data")
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    dataset = ImageFolderIPC(root=args.distill_data_path, ipc=args.cpc, transform=train_transform)
        
    print("Loading validation data")
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    dataset_test = TinyImageNet(args.data_path, split='val', download=True, transform=val_transform)

    print("Loading original training data")
    dataset_og = TinyImageNet(args.data_path, split='train', download=True, transform=val_transform)
    images_og = [torch.unsqueeze(dataset_og[i][0], dim=0) for i in range(len(dataset_og))]
    labels_og = [dataset_og[i][1] for i in range(len(dataset_og))]
    images_og = torch.cat(images_og, dim=0)
    labels_og = torch.tensor(labels_og, dtype=torch.long)

    print("Creating data loaders")
    train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    return dataset, images_og, labels_og, dataset_test, train_sampler, test_sampler

def create_model(model_name, device, num_classes, path=None):
    model = torchvision.models.get_model(model_name, weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1), bias=False)
    model.maxpool = nn.Identity()
    if path is not None:
        checkpoint = torch.load(path, map_location="cpu")
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if "module." in list(checkpoint.keys())[0]:
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
        model.load_state_dict(checkpoint)
    model.to(device)
    return model

def curriculum_arrangement(spc, curriculum_num):
    remainder = spc % curriculum_num
    arrangement = [spc // curriculum_num] * curriculum_num
    for i in range(remainder):
        arrangement[i] += 1

    return arrangement

def train_one_epoch(model, teacher_model, criterion, optimizer, data_loader, device, epoch, args):
    model.train()
    teacher_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value}"))
    metric_logger.add_meter("img/s", utils.SmoothedValue(window_size=10, fmt="{value}"))

    header = f"Epoch: [{epoch}]"
    for i, (image, target) in enumerate(metric_logger.log_every(data_loader, args.print_freq, header)):
        start_time = time.time()
        image, target = image.to(device), target.to(device)
        teacher_output = teacher_model(image)
        output = model(image)
        teacher_output_log_softmax = F.log_softmax(teacher_output/args.temperature, dim=1)
        output_log_softmax = F.log_softmax(output/args.temperature, dim=1)
        loss = criterion(output_log_softmax, teacher_output_log_softmax) * (args.temperature ** 2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
        batch_size = image.shape[0]
        metric_logger.update(loss=loss.item(), lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
        metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
        metric_logger.meters["img/s"].update(batch_size / (time.time() - start_time))

def evaluate(model, criterion, data_loader, device, log_suffix=""):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = f"Test: {log_suffix}"

    num_processed_samples = 0
    with torch.inference_mode():
        for image, target in data_loader:
            image = image.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(image)
            loss = criterion(output, target)

            acc1, acc5 = utils.accuracy(output, target, topk=(1, 5))
            batch_size = image.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters["acc1"].update(acc1.item(), n=batch_size)
            metric_logger.meters["acc5"].update(acc5.item(), n=batch_size)
            num_processed_samples += batch_size

    num_processed_samples = utils.reduce_across_processes(num_processed_samples)
    if (
        hasattr(data_loader.dataset, "__len__")
        and len(data_loader.dataset) != num_processed_samples
        and torch.distributed.get_rank() == 0
    ):
        warnings.warn(
            f"It looks like the dataset has {len(data_loader.dataset)} samples, but {num_processed_samples} "
            "samples were used for the validation, which might bias the results. "
            "Try adjusting the batch size and / or the world size. "
            "Setting the world size to 1 is always a safe bet."
        )

    metric_logger.synchronize_between_processes()

    return metric_logger.acc1.global_avg

def curriculum_train(current_curriculum, dst_train, test_loader, model, teacher_model, args):
    best_acc1 = 0
    
    train_sampler = torch.utils.data.RandomSampler(dst_train)
    train_loader = torch.utils.data.DataLoader(
        dst_train,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=True,
    )
    
    criterion = nn.CrossEntropyLoss()
    criterion_kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
    
    parameters = utils.set_weight_decay(model, args.weight_decay)
    
    opt_name = args.opt.lower()
    if opt_name.startswith("sgd"):
        optimizer = torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov="nesterov" in opt_name,
        )
    elif opt_name == "rmsprop":
        optimizer = torch.optim.RMSprop(
            parameters, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, eps=0.0316, alpha=0.9
        )
    elif opt_name == "adamw":
        optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise RuntimeError(f"Invalid optimizer {args.opt}. Only SGD, RMSprop and AdamW are supported.")

    args.lr_scheduler = args.lr_scheduler.lower()
    if args.lr_scheduler == "steplr":
        main_lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    elif args.lr_scheduler == "cosineannealinglr":
        main_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.lr_warmup_epochs, eta_min=0.0
        )
    elif args.lr_scheduler == "exponentiallr":
        main_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        raise RuntimeError(
            f"Invalid lr scheduler '{args.lr_scheduler}'. Only StepLR, CosineAnnealingLR and ExponentialLR "
            "are supported."
        )

    if args.lr_warmup_epochs > 0:
        if args.lr_warmup_method == "linear":
            warmup_lr_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        elif args.lr_warmup_method == "constant":
            warmup_lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, factor=args.lr_warmup_decay, total_iters=args.lr_warmup_epochs
            )
        else:
            raise RuntimeError(
                f"Invalid warmup lr method '{args.lr_warmup_method}'. Only linear and constant are supported."
            )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_lr_scheduler, main_lr_scheduler], milestones=[args.lr_warmup_epochs]
        )
    else:
        lr_scheduler = main_lr_scheduler
        
    print("Start training on synthetic dataset...")
    start_time = time.time()
    pbar = tqdm(range(args.epochs), ncols=100)
    for epoch in pbar:
        train_one_epoch(model, teacher_model, criterion_kl, optimizer, train_loader, args.device, epoch, args)
        lr_scheduler.step()
        if epoch > args.epochs * 0.8:
            acc1 = evaluate(model, criterion, test_loader, device=args.device)
            if acc1 > best_acc1:
                best_acc1 = acc1
            pbar.set_description(f"Epoch[{epoch}] Test Acc: {acc1:.2f}% Best Acc: {best_acc1:.2f}%")
    print(f"Best Accuracy {best_acc1:.2f}%")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print(f"Training time {total_time_str}")
    
    return model, best_acc1

def coarse_filtering(images_all, labels_all, filter, batch_size, args, get_correct=True):
    true_labels = labels_all.cpu()
    filter.eval()
    logits = None
    for select_times in range((len(images_all)+batch_size-1)//batch_size):
        current_data_batch = images_all[batch_size*select_times : batch_size*(select_times+1)].detach().to(args.device)
        batch_logits = filter(current_data_batch)
        if logits == None:
            logits = batch_logits.detach()
        else:
            logits = torch.cat((logits, batch_logits.detach()),0)
    predicted_labels = torch.argmax(logits, dim=1).cpu()
    target_indices = torch.where(true_labels == predicted_labels)[0] if get_correct else torch.where(true_labels != predicted_labels)[0]
    target_indices = target_indices.tolist()
    print('Acc on training set: {:.2f}%'.format(100*len(target_indices)/len(images_all) if get_correct else 100*(1-len(target_indices)/len(images_all))))
    return target_indices, logits

def selection_logits(selected_idx, teacher_correct_idx, images_all, labels_all, filter, args):
    batch_size = 256
    true_labels = labels_all.cpu()
    filter.eval()
    print('Coarse Filtering...')
    if args.select_misclassified:
        target_indices, logits = coarse_filtering(images_all, labels_all, filter, batch_size, args, get_correct=False)
    else:
        target_indices, logits = coarse_filtering(images_all, labels_all, filter, batch_size, args, get_correct=True)
    
    if teacher_correct_idx is not None:
        target_indices = list(set(teacher_correct_idx) & set(target_indices) - set(selected_idx))
    else:
        target_indices = list(set(target_indices) - set(selected_idx))
    print('Fine Selection...')
    selection = []
    if args.balance:
        target_idx_per_class = [[] for c in range(args.num_classes)]
        for idx in target_indices:
            target_idx_per_class[true_labels[idx]].append(idx)
        for c in range(args.num_classes):
            if args.select_method == 'random':
                selection += random.sample(target_idx_per_class[c], args.curpc)
            elif args.select_method == 'hard':
                selection += sorted(target_idx_per_class[c], key=lambda i: logits[i][c], reverse=False)[:args.curpc]
            elif args.select_method == 'simple':
                selection += sorted(target_idx_per_class[c], key=lambda i: logits[i][c], reverse=True)[:args.curpc]
    else:
        if args.select_method == 'random':
            selection = random.sample(target_indices, args.curpc*args.num_classes)
        elif args.select_method == 'hard':
            selection = sorted(target_indices, key=lambda i: logits[i][true_labels[i]], reverse=False)[:args.curpc*args.num_classes]
        elif args.select_method == 'simple':
            selection = sorted(target_indices, key=lambda i: logits[i][true_labels[i]], reverse=True)[:args.curpc*args.num_classes]

    return selection

def selection_score(selected_idx, teacher_correct_idx, images_all, labels_all, filter, score, reverse, args):
    batch_size = 256
    true_labels = labels_all.cpu()
    filter.eval()
    
    print('Coarse Filtering...')
    if args.select_misclassified:
        target_indices, _ = coarse_filtering(images_all, labels_all, filter, batch_size, args, get_correct=False)
    else:
        target_indices, _ = coarse_filtering(images_all, labels_all, filter, batch_size, args, get_correct=True)
    
    if teacher_correct_idx is not None:
        target_indices = list(set(teacher_correct_idx) & set(target_indices) - set(selected_idx))
    else:
        target_indices = list(set(target_indices) - set(selected_idx))
    print('Fine Selection...')
    selection = []
    if args.balance:
        target_idx_per_class = [[] for c in range(args.num_classes)]
        for idx in target_indices:
            target_idx_per_class[true_labels[idx]].append(idx)
        for c in range(args.num_classes):
            if args.select_method == 'random':
                selection += random.sample(target_idx_per_class[c], min(args.curpc, len(target_idx_per_class[c])))
            elif args.select_method == 'hard':
                selection += sorted(target_idx_per_class[c], key=lambda i: score[i], reverse=reverse)[:args.curpc]
            elif args.select_method == 'simple':
                selection += sorted(target_idx_per_class[c], key=lambda i: score[i], reverse=not reverse)[:args.curpc]
    else:
        if args.select_method == 'random':
            selection = random.sample(target_indices, min(args.curpc*args.num_classes, len(target_indices)))
        elif args.select_method == 'hard':
            selection = sorted(target_indices, key=lambda i: score[i], reverse=reverse)[:args.curpc*args.num_classes]
        elif args.select_method == 'simple':
            selection = sorted(target_indices, key=lambda i: score[i], reverse=not reverse)[:args.curpc*args.num_classes]
            
    return selection

def main(args):
    '''Preparation'''
    print('=> args.output_dir', args.output_dir)
    start_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    log_dir = os.path.join(args.output_dir, 'Tiny', start_time)
    os.makedirs(log_dir, exist_ok=True)
    
    device = torch.device(args.device)
    if device.type == 'cuda':
        print('Using GPU')
        torch.backends.cudnn.benchmark = True
    args.cpc = int(args.image_per_class * args.alpha)   # condensed images per class
    args.spc = args.image_per_class - args.cpc          # selected real images per class
    args.num_classes = 200
    print('Target IPC: {}, num_classes: {}, distillation portion: {}, distilled images per class: {}, real images to be selected per class: {}'
        .format(args.image_per_class, args.num_classes, args.alpha, args.cpc, args.spc))
    dataset_dis, images_og, labels_og, dataset_test, train_sampler, test_sampler = load_data(args)
    if args.score == 'forgetting':
        score = np.load(args.score_path)
        reverse = True
    curriculum_num = args.curriculum_num
    arrangement = curriculum_arrangement(args.spc, curriculum_num)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=256, sampler=test_sampler, num_workers=args.workers, pin_memory=True)

    teacher_model = create_model(args.teacher_model, device, args.num_classes, args.teacher_path)
    for p in teacher_model.parameters():
        p.requires_grad = False
    teacher_model.eval()
    teacher_correct_idx, _ = coarse_filtering(images_og, labels_og, teacher_model, 256, args, get_correct=True)
    print('teacher acc@1 on original training data: {:.2f}%'.format(100*len(teacher_correct_idx)/len(images_og)))

    '''Curriculum selection'''
    idx_selected = []
    dataset_sel = None
    dst_sel_transform = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
    ])
    print('Selected images per class arrangement in each curriculum: ', arrangement)

    for i in range(curriculum_num):
        print('----Curriculum [{}/{}]----'.format(i+1, curriculum_num))
        args.curpc = arrangement[i]
        if i == 0:
            print('Begin with distilled dataset')
            syn_dataset = dataset_dis
            dataset_sel = []
        
        print('Synthetic dataset size:', len(syn_dataset), "distilled data:", len(dataset_dis), "selected data:", len(dataset_sel))
        filter = create_model(args.filter_model, device, args.num_classes)
        filter, best_acc1 = curriculum_train(i, syn_dataset, test_loader, filter, teacher_model, args)

        print('Selecting real data...')
        if args.score == 'logits':
            selection = selection_logits(idx_selected, teacher_correct_idx, images_og, labels_og, filter, args)
        else:
            selection = selection_score(idx_selected, teacher_correct_idx, images_og, labels_og, filter, score, reverse, args)
        idx_selected += selection
        print('Selected {} in this curriculum'.format(len(selection)))
        imgs_select = images_og[idx_selected]
        labs_select = labels_og[idx_selected]
        dataset_sel = utils.TensorDataset(imgs_select, labs_select, dst_sel_transform)
        syn_dataset = torch.utils.data.ConcatDataset([dataset_dis, dataset_sel])
    print('----All curricula finished----')
    print('Final synthetic dataset size:', len(syn_dataset), "distilled data:", len(dataset_dis), "selected data:", len(dataset_sel))   

    print('Saving selected indices...')
    idx_file = os.path.join(log_dir, f'selected_indices.json')
    with open(idx_file, 'w') as f:
        json.dump({'ipc': args.image_per_class,
                   'alpha': args.alpha, 
                   'idx_selected': idx_selected}, f)
    f.close()

    '''Final evaluation'''
    num_eval = args.num_eval
    accs = []
    for i in range(num_eval):
        print(f'Evaluation {i+1}/{num_eval}')
        eval_model = create_model(args.eval_model, device, args.num_classes)
        _, best_acc1 = curriculum_train(0, syn_dataset, test_loader, eval_model, teacher_model, args)
        accs.append(best_acc1)
    acc_mean = np.mean(accs)
    acc_std = np.std(accs)
    print('----Evaluation Results----')
    print(f'Acc@1(mean): {acc_mean:.2f}%, std: {acc_std:.2f}')

    print('Saving results...')
    log_file = os.path.join(log_dir, f'exp_log.txt')
    with open(log_file, 'w') as f:
        f.write('EXP Settings: \n')
        f.write(f'IPC: {args.image_per_class},\tdistillation portion: {args.alpha},\tcurriculum_num: {args.curriculum_num}\n')
        f.write(f'filter model: {args.filter_model},\tteacher model: {args.teacher_model},\tbatch_size: {args.batch_size},\tepochs: {args.epochs}\n')
        f.write(f'coarse stage strategy: {'select misclassified' if args.select_misclassified else 'select correctly classified'}\n')
        f.write(f'fine stage strategy: {args.select_method},\tdifficulty score: {args.score},\tbalance: {args.balance}\n')
        f.write(f'eval model: {args.eval_model},\tAcc@1: {acc_mean:.2f}%,\tstd: {acc_std:.2f}\n')
    f.close()

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)