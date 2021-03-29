import argparse
import torch.backends.cudnn as cudnn
import torch.utils.data
from datasets import AINetDataset

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from mrcnn_engine import train_one_epoch, evaluate
import mrcnn_utils

import torch.utils.data


def get_instance_segmentation_model(num_classes, pretrainded=True):
    # load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=pretrainded)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default="./data.json",
                        help="데이터셋 경로     ")
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="모델 체크포인트 경로")
    parser.add_argument('--batch_size', type=int, default=6,
                        help="배치 사이즈")
    parser.add_argument('--iterations', type=int, default=120000,
                        help="학습 반복 횟수")
    parser.add_argument('--workers', type=int, default=4,
                        help="데이터 로딩 스레드의 수")
    parser.add_argument('--print_freq', type=int, default=200,
                        help="학습 상태 출력 주기")
    parser.add_argument('--lr', type=float, default=1e-6,
                        help="학습율")
    parser.add_argument('--decay_lr_at', type=list, default=[60000, 80000, 100000],
                        help="학습율 감소 시점")
    parser.add_argument('--decay_lr_to', type=float, default=0.1,
                        help="학습율 감소량")
    parser.add_argument('--momentum', type=float, default=0.9,
                        help="역전파 모멘텀")
    parser.add_argument('--weight_decay', type=int, default=5e-4,
                        help="역전파 시 가중치 감소율")
    parser.add_argument('--cuda', type=int, default=0,
                        help="그래픽 카드 번호")
    args = parser.parse_args()

    # Data parameters
    data_folder = args.data_folder  # folder with data files
    keep_difficult = True  # use objects considered difficult to detect?

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Learning parameters
    checkpoint = args.checkpoint
    batch_size = args.batch_size
    iterations = args.iterations
    workers = args.workers
    print_freq = args.print_freq
    lr = args.lr
    decay_lr_at = args.decay_lr_at
    decay_lr_to = args.decay_lr_to
    momentum = args.momentum
    weight_decay = args.weight_decay
    grad_clip = None  # clip if gradients are exploding, which may happen at larger batch sizes (sometimes at 32) - you will recognize it by a sorting error in the MuliBox loss calculation

    cudnn.benchmark = True

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    num_classes = 3

    # move model to the right device
    # get the model using our helper function
    model = get_instance_segmentation_model(num_classes)

    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0005)

    # and a learning rate scheduler which decreases the learning rate by
    # 10x every 3 epochs
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    # Custom dataloaders
    train_dataset = AINetDataset(data_folder,
                                 split='train',
                                 model_name='mrcnn',
                                 keep_difficult=keep_difficult)
    valid_dataset = AINetDataset(data_folder,
                                 split='test',
                                 model_name='mrcnn',
                                 keep_difficult=keep_difficult)

    # indices = torch.randperm(len(train_dataset)).tolist()
    # temp_train = torch.utils.data.Subset(train_dataset, indices[:10])
    # indices = torch.randperm(len(valid_dataset)).tolist()
    # temp_valid = torch.utils.data.Subset(valid_dataset, indices[:10])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True,
                                               collate_fn=mrcnn_utils.collate_fn, num_workers=workers,
                                               pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=1, shuffle=False,
                                               collate_fn=mrcnn_utils.collate_fn, num_workers=workers,
                                               pin_memory=True)

    max_epochs = 10

    # Epochs
    for epoch in range(max_epochs):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=10)

        # update the learning rate
        lr_scheduler.step()

        torch.save(model.state_dict(), 'ckpt/mrcnn_epoch_{}.pth'.format(epoch))

        # evaluate on the test dataset
        evaluate(model, valid_loader, device=device)
