from torch.autograd import Variable
import argparse
import os
from metric import SegmentationMetric
from BiSeNet.build_BiSeNet import BiSeNet
import torch
from tensorboardX import SummaryWriter
import tqdm
import numpy as np
from utils import poly_lr_scheduler
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, \
    per_class_iu
from loss import DiceLoss
from loading_data import loading_data
from PIL import Image
import pandas as pd


def print_size_of_model(model):
    """ Prints the real size of the model """
    torch.save(model.state_dict(), "quantized.pth")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)

def quantization(val_loader, model, metric):
    metric.reset()
    model.eval()
    model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')

    torch.ao.quantization.fuse_modules(model.context_path.features, ['conv1', 'bn1', 'relu'], inplace=True)
    torch.ao.quantization.fuse_modules(model.context_path.features.layer1, ['1.conv1', '1.bn1', '1.relu'], inplace=True)
    torch.ao.quantization.fuse_modules(model.context_path.features.layer2, ['1.conv1', '1.bn1', '1.relu'], inplace=True)
    torch.ao.quantization.fuse_modules(model.context_path.features.layer3, ['1.conv1', '1.bn1', '1.relu'], inplace=True)
    torch.ao.quantization.fuse_modules(model.context_path.features.layer4, ['1.conv1', '1.bn1', '1.relu'], inplace=True)

    model_fp32_prepared = torch.ao.quantization.prepare(model)
    model_fp32_prepared = model_fp32_prepared.to('cpu')
    list_pixAcc = []
    list_mIoU = []
    list_loss = []
    i = 0
    model_int8 = torch.ao.quantization.convert(model_fp32_prepared)
    print_size_of_model(model_int8)
    validate(val_loader, model_int8)

def validate(val_loader, net):
    net.eval()
    iou_ = 0.0
    num_classes = 5
    iou_sum_classes = [0.0] * num_classes
    label_trues, label_preds = [], []
    net.cuda()
    for vi, data in enumerate(val_loader, 0):
        inputs, labels = data
        inputs = Variable(inputs, volatile=True).cuda()
        labels = Variable(labels, volatile=True).cuda().long()
        outputs = net(inputs)
        # for binary classification

        label_trues.append(labels.cpu().numpy())
        label_preds.append(outputs.argmax(dim=1).cpu().numpy())
            #if vi < 10:  # only do this for the first 10 examples
              #pred_mask = outputs.argmax(dim=1).cpu().numpy()
              #colored_mask = colorize_mask(pred_mask[0])  # colorize the first mask in the batch
              #colored_mask.show()  # display the mask
    print(scores(label_trues, label_preds, num_classes))

def _fast_hist(label_true, label_pred, n_class):
    mask = (label_true >= 0) & (label_true < n_class)
    hist = np.bincount(
        n_class * label_true[mask].astype(int) +
        label_pred[mask], minlength=n_class**2).reshape(n_class, n_class)
    return hist


def scores(label_trues, label_preds, n_class):
    hist = np.zeros((n_class, n_class))
    for lt, lp in zip(label_trues, label_preds):
        hist += _fast_hist(lt.flatten(), lp.flatten(), n_class)
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    acc_cls = np.nanmean(acc_cls)
    iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
    mean_iu = np.nanmean(iu)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    y_miou.append(mean_iu)
    y_acc.append(acc_cls)


    return {'Overall Acc: \t': acc,
            'Mean Acc : \t': acc_cls,
            'FreqW Acc : \t': fwavacc,
            'Mean IoU : \t': mean_iu,
            'Classwise Mean IoU': iu}

def validation(val_loader, model, criterion, metric):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric.reset()
    model.eval()
    list_pixAcc = []
    list_mIoU = []
    list_loss = []
    for inputs, masks in val_loader:
        inputs, masks = inputs.to(device), masks.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            loss = criterion(torch.squeeze(outputs), masks)
        metric.update(outputs, masks)
        pixAcc, mIoU = metric.get()
        list_pixAcc.append(pixAcc)
        list_mIoU.append(mIoU)
        list_loss.append(loss.item())
    average_pixAcc = sum(list_pixAcc) / len(list_pixAcc)
    average_mIoU = sum(list_mIoU) / len(list_mIoU)
    average_loss = sum(list_loss) / len(list_loss)
    print(f"Average loss: {average_loss}, Average mIoU: {average_mIoU}, Average pixAcc: {average_pixAcc}")


def train(args, model, optimizer, dataloader_train, dataloader_val, metric, segmentation_type):
    writer = SummaryWriter(comment=''.format(args.optimizer, args.context_path))
    weights = [0.5, 0.7, 0.7, 0.9, 0.7]
    #weights = [0.3, 0.6, 0.6, 0.8, 0.6]
    class_weights = torch.FloatTensor(weights).cuda()
    if args.loss == 'dice':
        loss_func = DiceLoss()
    elif args.loss == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss(weight=class_weights).cuda()
    max_miou = 0
    step = 0
    for epoch in range(args.num_epochs):
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description('epoch %d, lr %f' % (epoch, lr))
        loss_record = []
        if epoch >=3:
          model.apply(torch.quantization.disable_observer)
        for i, (data, label) in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()
            output, output_sup1, output_sup2 = model(data)
            output = torch.squeeze(output)
            output_sup1 = torch.squeeze(output_sup1)
            output_sup2 = torch.squeeze(output_sup2)
            loss1 = loss_func(output, label)
            loss2 = loss_func(output_sup1, label)
            loss3 = loss_func(output_sup2, label)
            loss = loss1 + loss2 + loss3
            tq.update(args.batch_size)
            tq.set_postfix(loss='%.6f' % loss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            writer.add_scalar('loss_step', loss, step)
            loss_record.append(loss.item())
        tq.close()
        loss_train_mean = np.mean(loss_record)
        writer.add_scalar('epoch/loss_epoch_train', float(loss_train_mean), epoch)
        print('loss for train : %f' % (loss_train_mean))
        y_loss.append(loss_train_mean)
        validate(dataloader_val, model)

        if epoch % args.validation_step == 0:

            if segmentation_type == "binary":
                validation(dataloader_val, model, loss_func, metric)
            else:
                validate(dataloader_val, model)


def main(params):
    # basic parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_epochs', type=int, default=3, help='Number of epochs to train for')
    parser.add_argument('--epoch_start_i', type=int, default=0, help='Start counting epochs from this number')
    parser.add_argument('--checkpoint_step', type=int, default=1, help='How often to save checkpoints (epochs)')
    parser.add_argument('--validation_step', type=int, default=1, help='How often to perform validation (epochs)')
    parser.add_argument('--dataset', type=str, default="CamVid", help='Dataset you are using.')
    parser.add_argument('--crop_height', type=int, default=720, help='Height of cropped/resized input image to network')
    parser.add_argument('--crop_width', type=int, default=720, help='Width of cropped/resized input image to network')
    parser.add_argument('--batch_size', type=int, default=1, help='Number of images in each batch')
    parser.add_argument('--context_path', type=str, default="resnet101",
                        help='The context path model you are using, resnet18, resnet101.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate used for train')
    parser.add_argument('--data', type=str, default='', help='path of training data')
    parser.add_argument('--num_workers', type=int, default=4, help='num of workers')
    parser.add_argument('--num_classes', type=int, default=2, help='num of object classes (with void)')  # BINARY
    parser.add_argument('--cuda', type=str, default='0', help='GPU ids used for training')
    parser.add_argument('--use_gpu', type=bool, default=True, help='whether to user gpu for training')
    parser.add_argument('--pretrained_model_path', type=str, default=None, help='path to pretrained model')
    parser.add_argument('--save_model_path', type=str, default=None, help='path to save model')
    parser.add_argument('--optimizer', type=str, default='rmsprop', help='optimizer, support rmsprop, sgd, adam')
    parser.add_argument('--loss', type=str, default='crossentropy', help='loss function, dice or crossentropy')

    args = parser.parse_args(params)

    dataloader_train, dataloader_val, restore_transform = loading_data()
    # build model
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    model = BiSeNet(args.num_classes, args.context_path).cuda()

    model = torch.nn.DataParallel(model)

    # build optimizer
    if args.optimizer == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    else:  # rmsprop
        print('not supported optimizer \n')
        return None

    # load pretrained model if exists
    if args.pretrained_model_path is not None:
        print('load model from %s ...' % args.pretrained_model_path)
        model.module.load_state_dict(torch.load(args.pretrained_model_path))
        print('Done!')

    # train and validation
    metric = SegmentationMetric(5)
    segmentation_type = "instance" #or binary
    train(args, model, optimizer, dataloader_train, dataloader_val, metric, segmentation_type)
    #torch.save(model.state_dict(), "bisenet.pth")
    quantization(dataloader_val, model, metric)


if __name__ == '__main__':
    params = [
        '--num_epochs', '20',
        '--learning_rate', '2.5e-2',
        '--data', '/path/to/CamVid',
        '--num_workers', '5',
        '--num_classes', '5',
        '--cuda', '0',
        '--batch_size', '4',  # 6 for resnet101, 12 for resnet18
        '--save_model_path', './checkpoints_18_sgd',
        '--context_path', 'resnet18',  # only support resnet18 and resnet101
        '--optimizer', 'sgd',

    ]
    main(params)
