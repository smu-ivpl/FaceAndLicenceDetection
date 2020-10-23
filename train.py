import argparse
import time
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from tqdm import tqdm
from model import SSD300, MultiBoxLoss
from datasets import AINetDataset
from utils import *
from eval import evaluate

parser = argparse.ArgumentParser()
parser.add_argument('--data_folder',    type=str,       default="./data.json",
                    help="데이터셋 경로     ")
parser.add_argument('--checkpoint',     type=str,       default=None,
                    help="모델 체크포인트 경로")
parser.add_argument('--batch_size',     type=int,       default=16,
                    help="배치 사이즈")
parser.add_argument('--iterations',     type=int,       default=120000,
                    help="학습 반복 횟수")
parser.add_argument('--workers',        type=int,       default=12,
                    help="데이터 로딩 스레드의 수")
parser.add_argument('--print_freq',     type=int,       default=200,
                    help="학습 상태 출력 주기")
parser.add_argument('--lr',             type=float,     default=1e-3,
                    help="학습율")
parser.add_argument('--decay_lr_at',    type=list,      default=[60000, 80000, 100000],
                    help="학습율 감소 시점")
parser.add_argument('--decay_lr_to',    type=float,     default=0.1,
                    help="학습율 감소량")
parser.add_argument('--momentum',       type=float,     default=0.9,
                    help="역전파 모멘텀")
parser.add_argument('--weight_decay',   type=int,       default=5e-4,
                    help="역전파 시 가중치 감소율")
parser.add_argument('--cuda',           type=int,       default=0,
                    help="그래픽 카드 번호")
args = parser.parse_args()

# Data parameters
data_folder = args.data_folder  # folder with data files
keep_difficult = True  # use objects considered difficult to detect?

# Model parameters
# Not too many here since the SSD300 has a very specific structure
n_classes = len(label_map)  # number of different types of objects
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Learning parameters
checkpoint      = args.checkpoint
batch_size      = args.batch_size
iterations      = args.iterations
workers         = args.workers
print_freq      = args.print_freq
lr              = args.lr
decay_lr_at     = args.decay_lr_at
decay_lr_to     = args.decay_lr_to
momentum        = args.momentum
weight_decay    = args.weight_decay
grad_clip       = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

cudnn.benchmark = True

def main():
    """
    Training.
    """
    global start_epoch, label_map, epoch, checkpoint, decay_lr_at

    # Initialize model or load checkpoint
    if checkpoint is None:
        start_epoch = 0
        model = SSD300(n_classes=n_classes)
        # Initialize the optimizer, with twice the default learning rate for biases, as in the original Caffe repo
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        # optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
        #                             lr=lr, momentum=momentum, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                     lr=lr, weight_decay=weight_decay)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        print('\nLoaded checkpoint from epoch %d.\n' % start_epoch)
        model = checkpoint['model']
        # optimizer = checkpoint['optimizer']
        biases = list()
        not_biases = list()
        for param_name, param in model.named_parameters():
            if param.requires_grad:
                if param_name.endswith('.bias'):
                    biases.append(param)
                else:
                    not_biases.append(param)
        # optimizer = torch.optim.SGD(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
        #                             lr=lr, momentum=momentum, weight_decay=weight_decay)
        optimizer = torch.optim.Adam(params=[{'params': biases, 'lr': 2 * lr}, {'params': not_biases}],
                                     lr=lr, weight_decay=weight_decay)

    # Move to default device
    model = model.to(device)
    criterion = MultiBoxLoss(priors_cxcy=model.priors_cxcy).to(device)

    # Custom dataloaders
    train_dataset = AINetDataset(data_folder,
                                 split='train',
                                 keep_difficult=keep_difficult)
    valid_dataset = AINetDataset(data_folder,
                                 split='test',
                                 keep_difficult=keep_difficult)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               collate_fn=train_dataset.collate_fn, num_workers=workers, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                               collate_fn=valid_dataset.collate_fn, num_workers=workers, pin_memory=True)

    epochs = iterations // (len(train_dataset) // batch_size)

    if start_epoch >= epochs:
        start_epoch = 0
        print("Recovered epoch number reset to 0")

    decay_lr_at = [it // (len(train_dataset) // batch_size) for it in decay_lr_at]

    best_APs = 0.
    best_mAP = 0.

    # Epochs
    for epoch in range(start_epoch, epochs):

        # Decay learning rate at particular epochs
        if epoch in decay_lr_at:
            adjust_learning_rate(optimizer, decay_lr_to)

        model.train()  # training mode enables dropout

        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch,
              max_epochs=epochs)

        APs, mAP = evaluate(test_loader=valid_loader,
                            model=model)

        print('Epoch: {}, Current: [{}/{}], Best: [{}/{}]'.format(epoch + 1, APs, mAP, best_APs, best_mAP))

        if mAP > best_mAP:
            best_APs = APs
            best_mAP = mAP

            # Save checkpoint
            save_checkpoint(epoch, model, optimizer)

def train(train_loader, model, criterion, optimizer, epoch, max_epochs):
    with tqdm(total=len(train_loader)) as t:
        t.set_description('Epoch: [{0}/{1}]'.format(epoch + 1, max_epochs))

        batch_time = AverageMeter()  # forward prop. + back prop. time
        data_time = AverageMeter()  # data loading time
        losses = AverageMeter()  # loss

        start = time.time()

        # Batches
        for i, (images, boxes, labels, _) in enumerate(train_loader):

            data_time.update(time.time() - start)

            # Move to default device
            images = images.to(device)  # (batch_size (N), 3, 300, 300)
            boxes = [b.to(device) for b in boxes]
            labels = [l.to(device) for l in labels]

            # Forward prop.
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)

            # Loss
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

            # Backward prop.
            optimizer.zero_grad()
            loss.backward()

            # Clip gradients, if necessary
            if grad_clip is not None:
                clip_gradient(optimizer, grad_clip)

            # Update model
            optimizer.step()

            losses.update(loss.item(), images.size(0))
            batch_time.update(time.time() - start)

            t.set_postfix(batch_time='{batch_time.val:.3f} ({batch_time.avg:.3f})'.format(batch_time=batch_time),
                          data_time='{data_time.val:.3f} ({data_time.avg:.3f})'.format(data_time=data_time),
                          loss='{loss.val:.4f} ({loss.avg:.4f})'.format(loss=losses))
            t.update(1)

            start = time.time()

        del predicted_locs, predicted_scores, images, boxes, labels  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()
