import argparse
import os

import numpy as np
import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict, load_checkpoint_adv
from mmcv.parallel import scatter, collate, MMDataParallel

from mmaction import datasets
from mmaction.datasets import build_dataloader
from mmaction.models import build_recognizer, recognizers
from mmaction.core.evaluation.accuracy import (softmax, top_k_accuracy,
                                               mean_class_accuracy)

import pickle
import os.path as osp
import pdb


def single_test(model, data_loader, cfg):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            #result, seg, img, meta = model(return_loss=False, **data)
            result = model(return_loss=False, **data)
            '''
            for frame in range(8):
                for idx_in_batch in range(seg.shape[0]):
                    seg0 = seg.argmax(1)[idx_in_batch,frame,:,:]
                    img0 = mmcv.imdenormalize(img[idx_in_batch,:,frame*4,:,:].transpose([1,2,0]),
                            mean=np.array(cfg.img_norm_cfg.mean).reshape(1,1,3),
                            std=np.array(cfg.img_norm_cfg.std).reshape(1,1,3),
                            to_bgr=cfg.img_norm_cfg.to_rgb)

                    out_dir = os.path.join('outputs_ntucentercrop', meta[0]['img_path'], 'setting_%02d'%idx_in_batch)
                    if not os.path.isdir(out_dir):
                        os.makedirs(out_dir)
                    mmcv.imwrite(img0, os.path.join(out_dir, 'img_%05d.png'%(frame)))
                    mmcv.imwrite(seg0*255, os.path.join(out_dir, 'seg_%05d.png'%(frame)))
            '''
        results.append(result)

        batch_size = data['img_group_0'].data[0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--use_softmax', action='store_true',
                        help='whether to use softmax score')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    if cfg.data.test.oversample == 'three_crop':
        cfg.model.spatial_temporal_module.spatial_size = 8

    dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    cfg.model.gpus = args.gpus
    cfg.model.dist = False
    cfg.model.train = False
    if args.gpus == 1:
        model = build_recognizer(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint_adv(model, args.checkpoint, strict=False)
        model = MMDataParallel(model, device_ids=[0])

        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False)
        outputs = single_test(model, data_loader, cfg)
    else:
        model_args = cfg.model.copy()
        model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
        model_type = getattr(recognizers, model_args.pop('type'))
        outputs = parallel_test(
            model_type,
            model_args,
            args.checkpoint,
            dataset,
            _data_func,
            range(args.gpus),
            workers_per_gpu=args.proc_per_gpu)


    if args.out:
        print('writing results to {}'.format(args.out))
        mmcv.dump(outputs, args.out)

    gt_labels = []
    for i in range(len(dataset)):
        ann = dataset.get_ann_info(i)
        gt_labels.append(ann['label'])

    if args.use_softmax:
        print("Averaging score over {} clips with softmax".format(
            outputs[0].shape[0]))
        results = [softmax(res, dim=1).mean(axis=0) for res in outputs]
    else:
        print("Averaging score over {} clips without softmax (ie, raw)".format(
            outputs[0].shape[0]))
        results = [res.mean(axis=0) for res in outputs]

    import datetime

    currentDT = datetime.datetime.now()


    with open(osp.join(args.checkpoint + '.result_%s.pkl' % currentDT.strftime("%Y-%m-%d_%H:%M:%S")), 'wb' ) as f:
        pickle.dump([results, gt_labels], f)
    top1, top5 = top_k_accuracy(results, gt_labels, k=(1, 5))
    mean_acc = mean_class_accuracy(results, gt_labels)
    print("Mean Class Accuracy = {:.02f}".format(mean_acc * 100))
    print("Top-1 Accuracy = {:.02f}".format(top1 * 100))
    print("Top-5 Accuracy = {:.02f}".format(top5 * 100))


if __name__ == '__main__':
    main()
