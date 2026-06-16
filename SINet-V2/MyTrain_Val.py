# author: Daniel-Ji (e-mail: gepengai.ji@gmail.com)
# data: 2021-01-16
import json
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
from datetime import datetime
from torchvision.utils import make_grid
from lib.Network_Res2Net_GRA_NCD import Network
from utils.data_val import get_loader, test_dataset
from utils.utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from vis import EvaluationMetricsV2

DATASET_ROOT = os.path.join(PROJECT_ROOT, 'Dataset')
WORKSPACE_ROOT = os.path.join(DATASET_ROOT, 'RISE_Workspace')


def ensure_sep(path):
    return path if path.endswith(os.sep) else path + os.sep


def list_image_basenames(root):
    valid_suffixes = ('.jpg', '.jpeg', '.png', '.bmp')
    if not os.path.isdir(root):
        return set()
    return {
        os.path.basename(name)
        for name in os.listdir(root)
        if name.lower().endswith(valid_suffixes)
    }


def check_train_val_overlap(train_image_root, val_dataset_roots):
    train_names = list_image_basenames(train_image_root)
    overlap_report = {}
    for dataset_name, dataset_root in val_dataset_roots.items():
        val_image_root = os.path.join(dataset_root, 'Image')
        val_names = list_image_basenames(val_image_root)
        overlap = sorted(train_names & val_names)
        if overlap:
            overlap_report[dataset_name] = overlap
    return overlap_report


def save_training_state(model, optimizer, epoch, save_path, best_mode, best_metric_value, best_epoch, best_metrics, step):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_mode': best_mode,
        'best_metric_value': best_metric_value,
        'best_epoch': best_epoch,
        'best_metrics': best_metrics,
        'step': step,
    }
    torch.save(state, os.path.join(save_path, 'last_checkpoint.pth'))


def load_training_state(checkpoint_path, model, optimizer):
    checkpoint = torch.load(checkpoint_path)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return {
            'start_epoch': int(checkpoint['epoch']) + 1,
            'best_mode': checkpoint.get('best_mode', 'mae'),
            'best_metric_value': float(checkpoint.get('best_metric_value', 1e9)),
            'best_epoch': int(checkpoint.get('best_epoch', 0)),
            'best_metrics': checkpoint.get('best_metrics', {}),
            'step': int(checkpoint.get('step', 0)),
        }

    model.load_state_dict(checkpoint)
    return {
        'start_epoch': 1,
        'best_mode': None,
        'best_metric_value': None,
        'best_epoch': 0,
        'best_metrics': {},
        'step': 0,
    }


def prepare_reliability(reliability, mask, boundary_downweight=0.0, min_weight=0.05):
    if reliability is None:
        reliability = torch.ones_like(mask)
    elif reliability.shape[-2:] != mask.shape[-2:]:
        reliability = F.interpolate(reliability, size=mask.shape[-2:], mode='bilinear', align_corners=False)

    reliability = reliability.clamp(min=min_weight, max=1.0)
    if boundary_downweight > 0:
        boundary = torch.abs(F.avg_pool2d(mask, kernel_size=7, stride=1, padding=3) - mask)
        boundary = (boundary > 0.05).float()
        reliability = reliability * (1.0 - boundary_downweight * boundary)
        reliability = reliability.clamp(min=min_weight, max=1.0)
    return reliability


def structure_loss(pred, mask, reliability=None, boundary_downweight=0.0):
    """
    loss function (ref: F3Net-AAAI-2020)
    """
    reliability = prepare_reliability(reliability, mask, boundary_downweight=boundary_downweight)
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    weit = weit * reliability
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduce='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def noise_correction_loss(pred, mask, reliability=None, q=2.0, boundary_downweight=0.0):
    reliability = prepare_reliability(reliability, mask, boundary_downweight=boundary_downweight)
    pred = torch.sigmoid(pred)
    numerator = (torch.abs(pred - mask).clamp(min=1e-6).pow(q) * reliability).sum(dim=(2, 3))
    denominator = ((pred + mask - pred * mask) * reliability).sum(dim=(2, 3)) + 1.0
    return (numerator / denominator).mean()


def robust_loss(pred, mask, reliability, epoch):
    base = structure_loss(pred, mask, reliability=reliability, boundary_downweight=opt.boundary_downweight)
    if opt.nc_weight <= 0:
        return base, torch.zeros((), device=pred.device)

    q_value = opt.nc_q_late if epoch >= opt.nc_q_switch else opt.nc_q_early
    correction = noise_correction_loss(
        pred,
        mask,
        reliability=reliability,
        q=q_value,
        boundary_downweight=opt.boundary_downweight,
    )
    return base + opt.nc_weight * correction, correction


def train(train_loader, model, optimizer, epoch, save_path, writer):
    """
    train function
    """
    global step
    model.train()
    loss_all = 0
    epoch_step = 0
    try:
        for i, batch in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            if len(batch) == 3:
                images, gts, reliability = batch
                reliability = reliability.cuda()
            else:
                images, gts = batch
                reliability = None

            images = images.cuda()
            gts = gts.cuda()

            preds = model(images)
            loss_0, nc_0 = robust_loss(preds[0], gts, reliability, epoch)
            loss_1, nc_1 = robust_loss(preds[1], gts, reliability, epoch)
            loss_2, nc_2 = robust_loss(preds[2], gts, reliability, epoch)
            loss_final, nc_final = robust_loss(preds[3], gts, reliability, epoch)
            loss_init = loss_0 + loss_1 + loss_2
            nc_loss = nc_0 + nc_1 + nc_2 + nc_final

            loss = loss_init + loss_final

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()

            step += 1
            epoch_step += 1
            loss_all += loss.data

            if i % 20 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} Loss2: {:0.4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_init.data,
                             loss_final.data))
                logging.info(
                    '[Train Info]:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Total_loss: {:.4f} Loss1: {:.4f} '
                    'Loss2: {:0.4f} NC: {:0.4f}'.
                    format(epoch, opt.epoch, i, total_step, loss.data, loss_init.data, loss_final.data, nc_loss.data))
                # TensorboardX-Loss
                writer.add_scalars('Loss_Statistics',
                                   {'Loss_init': loss_init.data, 'Loss_final': loss_final.data,
                                    'Loss_total': loss.data, 'Loss_nc': nc_loss.data},
                                   global_step=step)
                # TensorboardX-Training Data
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('GT', grid_image, step)

                # TensorboardX-Outputs
                res = preds[0][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_init', torch.tensor(res), step, dataformats='HW')
                res = preds[3][0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('Pred_final', torch.tensor(res), step, dataformats='HW')

        loss_all /= epoch_step
        logging.info('[Train Info]: Epoch [{:03d}/{:03d}], Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        save_training_state(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            save_path=save_path,
            best_mode=opt.best_mode,
            best_metric_value=best_metric_value,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            step=step,
        )
        if epoch % 50 == 0:
            torch.save(model.state_dict(), os.path.join(save_path, 'Net_epoch_{}.pth'.format(epoch)))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        save_training_state(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            save_path=save_path,
            best_mode=opt.best_mode,
            best_metric_value=best_metric_value,
            best_epoch=best_epoch,
            best_metrics=best_metrics,
            step=step,
        )
        torch.save(model.state_dict(), os.path.join(save_path, 'Net_epoch_{}.pth'.format(epoch + 1)))
        print('Save checkpoints successfully!')
        raise


def evaluate_loader(test_loader, model):
    metric = EvaluationMetricsV2()
    with torch.no_grad():
        for _ in range(test_loader.size):
            image, gt, name, img_for_post = test_loader.load_data()
            gt = np.array(gt)
            image = image.cuda()

            res = model(image)
            res = F.upsample(res[3], size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            metric.step(pred=res, gt=gt)
    metric_dic = metric.get_results()
    return {
        'sm': float(metric_dic['sm']),
        'wfm': float(metric_dic['wfm']),
        'mae': float(metric_dic['mae']),
        'emMean': float(metric_dic['emMean']),
    }


def val(test_loaders, model, epoch, save_path, writer):
    """
    validation function
    """
    global best_metric_value, best_epoch, best_metrics
    model.eval()
    with torch.no_grad():
        metrics = {}
        for dataset_name, test_loader in test_loaders.items():
            metrics[dataset_name] = evaluate_loader(test_loader, model)
            writer.add_scalar('Val/{}/Sm'.format(dataset_name), torch.tensor(metrics[dataset_name]['sm']),
                              global_step=epoch)
            writer.add_scalar('Val/{}/MAE'.format(dataset_name), torch.tensor(metrics[dataset_name]['mae']),
                              global_step=epoch)
            writer.add_scalar('Val/{}/wFm'.format(dataset_name), torch.tensor(metrics[dataset_name]['wfm']),
                              global_step=epoch)

        if opt.best_mode == 'mae':
            primary_dataset = next(iter(metrics))
            metric_value = metrics[primary_dataset]['mae']
            is_better = metric_value < best_metric_value
            metric_desc = '{} mae {:.6f}'.format(primary_dataset, metric_value)
        elif opt.best_mode == 'joint_sm_mae':
            sm_mean = float(np.mean([item['sm'] for item in metrics.values()]))
            mae_mean = float(np.mean([item['mae'] for item in metrics.values()]))
            metric_value = sm_mean - mae_mean
            is_better = metric_value > best_metric_value
            writer.add_scalar('Val/joint_sm_mae', torch.tensor(metric_value), global_step=epoch)
            metric_desc = 'joint_sm_mae {:.6f} (mean_sm {:.6f}, mean_mae {:.6f})'.format(
                metric_value, sm_mean, mae_mean
            )
        else:
            dataset_scores = {}
            for dataset_name, item in metrics.items():
                dataset_scores[dataset_name] = (
                    0.35 * item['sm'] +
                    0.15 * item['emMean'] +
                    0.30 * item['wfm'] +
                    0.20 * (1.0 - item['mae'])
                )

            score_values = list(dataset_scores.values())
            mean_score = float(np.mean(score_values))
            min_score = float(np.min(score_values))
            metric_value = 0.80 * mean_score + 0.20 * min_score
            is_better = metric_value > best_metric_value
            writer.add_scalar('Val/joint_four_metrics', torch.tensor(metric_value), global_step=epoch)
            metric_desc = 'joint_four_metrics {:.6f} (mean_score {:.6f}, min_score {:.6f})'.format(
                metric_value, mean_score, min_score
            )

        metrics_line = ' | '.join(
            '{} sm={:.4f} em={:.4f} mae={:.4f} wfm={:.4f}'.format(
                name, item['sm'], item['emMean'], item['mae'], item['wfm']
            )
            for name, item in metrics.items()
        )
        print('Epoch: {}, {}, bestMetric: {:.6f}, bestEpoch: {}. {}'.format(
            epoch, metric_desc, best_metric_value, best_epoch, metrics_line
        ))

        if is_better:
            best_metric_value = metric_value
            best_epoch = epoch
            best_metrics = metrics
            torch.save(model.state_dict(), os.path.join(save_path, 'Net_epoch_best.pth'))
            with open(os.path.join(save_path, 'best_metrics.json'), 'w', encoding='utf-8') as handle:
                json.dump({
                    'epoch': best_epoch,
                    'best_mode': opt.best_mode,
                    'best_metric_value': best_metric_value,
                    'metrics': best_metrics,
                    'dataset_scores': dataset_scores if opt.best_mode == 'joint_four_metrics' else None,
                }, handle, ensure_ascii=False, indent=2)
            print('Save state_dict successfully! Best epoch:{}.'.format(epoch))

        logging.info('[Val Info]: Epoch:{} {} {}'.format(epoch, metric_desc, metrics_line))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100, help='epoch number')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=352, help='training dataset size')
    parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
    parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
    parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
    parser.add_argument('--load', type=str, default=None, help='train from checkpoints')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')
    parser.add_argument('--img_root', type=str, default=os.path.join(DATASET_ROOT, 'TrainDataset', 'Image'),
                        help='the training rgb images root')
    parser.add_argument('--gt_root', type=str, default=os.path.join(WORKSPACE_ROOT, 'pseudo_mask'), )
    parser.add_argument('--val_root', type=str, default=os.path.join(DATASET_ROOT, 'TestDataset', 'CAMO'),
                        help='the test rgb images root')
    parser.add_argument('--test_dataset_root', type=str, default=os.path.join(DATASET_ROOT, 'TestDataset'),
                        help='root of benchmark datasets')
    parser.add_argument('--val_datasets', type=str, default='',
                        help='comma separated dataset names under test_dataset_root, e.g. CAMO,COD10K')
    parser.add_argument('--best_mode', type=str, default='mae', choices=['mae', 'joint_sm_mae', 'joint_four_metrics'],
                        help='criterion used to select best checkpoint')
    parser.add_argument('--use_reliability_map', action='store_true',
                        help='return an old/refined agreement map and use it as pixel-wise pseudo-label reliability')
    parser.add_argument('--old_mask_root', type=str, default='',
                        help='root of original pseudo masks used to build reliability maps')
    parser.add_argument('--refined_mask_root', type=str, default='',
                        help='root of refined pseudo masks used to build reliability maps')
    parser.add_argument('--disagreement_weight', type=float, default=0.35,
                        help='pixel weight assigned to old/refined disagreement regions')
    parser.add_argument('--boundary_downweight', type=float, default=0.0,
                        help='downweight pseudo-label boundary pixels inside the loss, 0 disables it')
    parser.add_argument('--disable_gt_pepper', action='store_true',
                        help='disable salt-and-pepper perturbation on pseudo masks')
    parser.add_argument('--nc_weight', type=float, default=0.0,
                        help='weight of noise-correction loss; 0 keeps the original training objective')
    parser.add_argument('--nc_q_early', type=float, default=2.0,
                        help='early-stage exponent for noise-correction loss')
    parser.add_argument('--nc_q_late', type=float, default=1.0,
                        help='late-stage exponent for noise-correction loss')
    parser.add_argument('--nc_q_switch', type=int, default=40,
                        help='epoch where noise-correction loss switches from nc_q_early to nc_q_late')
    parser.add_argument('--resume_last', action='store_true',
                        help='resume from save_path/last_checkpoint.pth if it exists')
    parser.add_argument('--resume_path', type=str, default='',
                        help='explicit checkpoint path to resume from')
    parser.add_argument('--save_path', type=str,
                        default='./snapshot/SINet_V2/baseline/',
                        help='the path to save model and log')
    parser.add_argument('--allow_val_overlap', action='store_true',
                        help='allow train/val image basename overlap; disabled by default to prevent leakage')
    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
    elif opt.gpu_id == '1':
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"
        print('USE GPU 1')
    cudnn.benchmark = True

    # build the model
    model = Network(channel=32).cuda()

    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)

    optimizer = torch.optim.Adam(model.parameters(), opt.lr)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # load data
    print('load data...')
    train_loader = get_loader(image_root=ensure_sep(opt.img_root),
                              gt_root=ensure_sep(opt.gt_root),
                              batchsize=opt.batchsize,
                              trainsize=opt.trainsize,
                              num_workers=4,
                              pin_memory=False,
                              old_mask_root=ensure_sep(opt.old_mask_root) if opt.old_mask_root else None,
                              refined_mask_root=ensure_sep(opt.refined_mask_root) if opt.refined_mask_root else None,
                              return_reliability=opt.use_reliability_map,
                              disagreement_weight=opt.disagreement_weight,
                              gt_pepper=not opt.disable_gt_pepper)
    if opt.val_datasets.strip():
        val_dataset_names = [name.strip() for name in opt.val_datasets.split(',') if name.strip()]
        val_dataset_roots = {}
        val_loaders = {}
        for dataset_name in val_dataset_names:
            dataset_root = os.path.join(opt.test_dataset_root, dataset_name)
            val_dataset_roots[dataset_name] = dataset_root
            val_loaders[dataset_name] = test_dataset(
                image_root=os.path.join(dataset_root, 'Image') + '/',
                gt_root=os.path.join(dataset_root, 'GT') + '/',
                testsize=opt.trainsize,
            )
    else:
        val_dataset_name = os.path.basename(os.path.normpath(opt.val_root))
        val_dataset_roots = {
            val_dataset_name: opt.val_root
        }
        val_loaders = {
            val_dataset_name: test_dataset(
                image_root=os.path.join(opt.val_root, 'Image') + '/',
                gt_root=os.path.join(opt.val_root, 'GT') + '/',
                testsize=opt.trainsize,
            )
        }

    if not opt.allow_val_overlap:
        overlap_report = check_train_val_overlap(opt.img_root, val_dataset_roots)
        if overlap_report:
            message_lines = ['detected train/val overlap by image basename:']
            for dataset_name, overlap in overlap_report.items():
                preview = ', '.join(overlap[:10])
                if len(overlap) > 10:
                    preview += ', ...'
                message_lines.append(
                    '{}: {} overlapping files ({})'.format(dataset_name, len(overlap), preview)
                )
            message_lines.append(
                'fix the split or pass --allow_val_overlap only if leakage is intentional'
            )
            raise ValueError('\n'.join(message_lines))
    total_step = len(train_loader)

    # logging
    logging.basicConfig(filename=os.path.join(save_path, 'log.log'),
                        format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                        level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')
    logging.info("Network-Train")
    logging.info('Config: epoch: {}; lr: {}; batchsize: {}; trainsize: {}; clip: {}; decay_rate: {}; load: {}; '
                 'save_path: {}; decay_epoch: {}; use_reliability_map: {}; nc_weight: {}; '
                 'boundary_downweight: {}; disagreement_weight: {}; disable_gt_pepper: {}'.format(
                     opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip,
                     opt.decay_rate, opt.load, save_path, opt.decay_epoch, opt.use_reliability_map,
                     opt.nc_weight, opt.boundary_downweight, opt.disagreement_weight, opt.disable_gt_pepper
                 ))

    step = 0
    writer = SummaryWriter(os.path.join(save_path, 'summary'))
    best_metric_value = 1 if opt.best_mode == 'mae' else -1e9
    best_epoch = 0
    best_metrics = {}
    start_epoch = 1

    resume_candidate = None
    if opt.resume_path:
        resume_candidate = opt.resume_path
    elif opt.resume_last:
        candidate = os.path.join(save_path, 'last_checkpoint.pth')
        if os.path.exists(candidate):
            resume_candidate = candidate

    if resume_candidate:
        resume_state = load_training_state(resume_candidate, model, optimizer)
        start_epoch = resume_state['start_epoch']
        step = resume_state['step']
        if resume_state['best_mode'] == opt.best_mode and resume_state['best_metric_value'] is not None:
            best_metric_value = resume_state['best_metric_value']
        best_epoch = resume_state['best_epoch']
        best_metrics = resume_state['best_metrics']
        print('resume training from {}, next epoch {}'.format(resume_candidate, start_epoch))
        logging.info('Resume from {} next epoch {}'.format(resume_candidate, start_epoch))

    print("Start train...")
    for epoch in range(start_epoch, opt.epoch + 1):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path, writer)
        val(val_loaders, model, epoch, save_path, writer)
