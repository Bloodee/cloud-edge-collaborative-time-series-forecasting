"""Command-line entry point for the CE-BiD main experiment."""

import argparse
import random

import numpy as np
import torch

from exp.exp_distill import Exp_Distill
from exp.exp_local_train import Exp_LocalTrain
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_ofa_distill import Exp_OFA_Distill
from exp.exp_reverse_distill import Exp_ReverseDistill


def build_parser():
    parser = argparse.ArgumentParser(
        description='CE-BiD cloud-edge collaborative time-series forecasting')

    basic = parser.add_argument_group('basic')
    basic.add_argument('--task_name', choices=['long_term_forecast'],
                       default='long_term_forecast')
    basic.add_argument('--is_training', type=int, choices=[0, 1], default=1)
    basic.add_argument('--model_id', default='ce_bid')
    basic.add_argument('--model', choices=['TimesNet', 'CNN'], default='TimesNet')
    basic.add_argument('--seed', type=int, default=2021)
    basic.add_argument('--des', default='experiment')
    basic.add_argument('--itr', type=int, default=1)

    data = parser.add_argument_group('data')
    data.add_argument('--data', choices=['custom'], default='custom')
    data.add_argument('--root_path', default='./dataset/merge/')
    data.add_argument('--data_path', default='PVOD.csv')
    data.add_argument('--features', choices=['M', 'S', 'MS'], default='M')
    data.add_argument('--target', default='OT')
    data.add_argument('--freq', default='h')
    data.add_argument('--seq_len', type=int, default=12)
    data.add_argument('--label_len', type=int, default=12)
    data.add_argument('--pred_len', type=int, default=6)
    data.add_argument('--specific_start', type=int)
    data.add_argument('--specific_end', type=int)
    data.add_argument('--train_ratio', type=float)
    data.add_argument('--val_ratio', type=float)
    data.add_argument('--test_ratio', type=float)
    data.add_argument('--step', type=int, default=0)
    data.add_argument('--total_steps', type=int, default=1)
    data.add_argument('--scaler_path')

    model = parser.add_argument_group('model')
    model.add_argument('--enc_in', type=int, default=15)
    model.add_argument('--dec_in', type=int, default=15)
    model.add_argument('--c_out', type=int, default=15)
    model.add_argument('--d_model', type=int, default=16)
    model.add_argument('--d_ff', type=int, default=32)
    model.add_argument('--e_layers', type=int, default=2)
    model.add_argument('--d_layers', type=int, default=1)
    model.add_argument('--n_heads', type=int, default=8)
    model.add_argument('--factor', type=int, default=3)
    model.add_argument('--top_k', type=int, default=5)
    model.add_argument('--num_kernels', type=int, default=6)
    model.add_argument('--dropout', type=float, default=0.1)
    model.add_argument('--embed', choices=['timeF', 'fixed', 'learned'],
                       default='timeF')

    training = parser.add_argument_group('training')
    training.add_argument('--checkpoints', default='./checkpoints/')
    training.add_argument('--num_workers', type=int, default=0)
    training.add_argument('--train_epochs', type=int, default=10)
    training.add_argument('--batch_size', type=int, default=32)
    training.add_argument('--patience', type=int, default=3)
    training.add_argument('--learning_rate', type=float, default=1e-4)
    training.add_argument('--lradj', choices=['type1', 'type2', 'type3', 'cosine'],
                          default='type1')
    training.add_argument('--use_amp', action='store_true')
    training.add_argument('--inverse', action='store_true')

    device = parser.add_argument_group('device')
    device.add_argument('--use_gpu', type=int, choices=[0, 1], default=1)
    device.add_argument('--gpu', type=int, default=0)
    device.add_argument('--gpu_type', choices=['cuda', 'mps'], default='cuda')
    device.add_argument('--use_multi_gpu', action='store_true')
    device.add_argument('--devices', default='0')

    distill = parser.add_argument_group('distillation')
    distill.add_argument('--do_distill', action='store_true')
    distill.add_argument('--do_distill_test', action='store_true')
    distill.add_argument('--do_local_train', action='store_true')
    distill.add_argument('--do_reverse_distill', action='store_true')
    distill.add_argument('--pretrained_model_path')
    distill.add_argument('--teacher_model_path')
    distill.add_argument('--pretrained_teacher_path')
    distill.add_argument('--student_model_path', nargs='+')
    distill.add_argument('--teacher_model_name', choices=['TimesNet'],
                         default='TimesNet')
    distill.add_argument('--student_model_name', choices=['CNN'], default='CNN')
    distill.add_argument('--cloud_dim', type=int, default=99)
    distill.add_argument('--teacher_d_model', type=int, default=128)
    distill.add_argument('--teacher_d_ff', type=int, default=256)
    distill.add_argument('--teacher_e_layers', type=int, default=2)
    distill.add_argument('--student_d_model', type=int, default=16)
    distill.add_argument('--student_d_ff', type=int, default=32)
    distill.add_argument('--student_e_layers', type=int, default=2)
    distill.add_argument('--student_d_layers', type=int, default=1)
    distill.add_argument('--lambda_kd', type=float, default=0.5)
    distill.add_argument('--ot_weight', type=float, default=1.0)
    distill.add_argument('--use_aux_kd', action='store_true')
    distill.add_argument('--aux_kd_weight', type=float, default=0.3)
    distill.add_argument('--use_ofa_kd', action='store_true')
    distill.add_argument('--ofa_eps', type=float, default=1.5)
    distill.add_argument('--ofa_loss_weight', type=float, default=0.5)
    distill.add_argument('--ofa_final_weight', type=float, default=0.5)
    distill.add_argument('--ofa_anchor_weight', type=float, default=0.3)
    distill.add_argument('--rev_kd_weight', type=float, default=0.3)
    distill.add_argument('--rev_weight_reg', type=float, default=0.01)
    distill.add_argument('--rev_rollback_thresh', type=float, default=1.1)
    distill.add_argument('--rev_gamma', type=float, default=1.5)
    return parser


def validate_args(parser, args):
    positive = (
        'seq_len', 'label_len', 'pred_len', 'enc_in', 'dec_in', 'c_out',
        'd_model', 'd_ff', 'e_layers', 'batch_size', 'train_epochs', 'itr',
        'total_steps', 'top_k', 'num_kernels', 'cloud_dim',
        'teacher_d_model', 'teacher_d_ff', 'teacher_e_layers',
        'student_d_model', 'student_d_ff', 'student_e_layers',
    )
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f'--{name} must be positive')
    if args.label_len > args.seq_len:
        parser.error('--label_len cannot exceed --seq_len')
    if args.step < 0 or args.step >= args.total_steps:
        parser.error('--step must satisfy 0 <= step < total_steps')
    if args.num_workers < 0 or args.gpu < 0:
        parser.error('num_workers and gpu cannot be negative')
    if args.patience <= 0:
        parser.error('--patience must be positive')
    if args.learning_rate <= 0:
        parser.error('--learning_rate must be positive')
    if not 0 <= args.dropout < 1:
        parser.error('--dropout must be in [0, 1)')
    if not 0 <= args.lambda_kd <= 1 or not 0 <= args.rev_kd_weight <= 1:
        parser.error('lambda_kd and rev_kd_weight must be in [0, 1]')
    if any(value < 0 for value in (
            args.ot_weight, args.aux_kd_weight, args.rev_weight_reg, args.ofa_eps,
            args.ofa_loss_weight, args.ofa_final_weight, args.ofa_anchor_weight)):
        parser.error('distillation weights and ofa_eps cannot be negative')
    if args.rev_rollback_thresh < 1 or args.rev_gamma <= 0:
        parser.error('rollback threshold must be >= 1 and rev_gamma positive')
    if (args.specific_start is None) != (args.specific_end is None):
        parser.error('specific_start and specific_end must be provided together')
    if args.specific_start is not None and args.specific_start >= args.specific_end:
        parser.error('specific_start must be smaller than specific_end')

    supplied = [args.train_ratio, args.val_ratio, args.test_ratio]
    if any(value is not None for value in supplied):
        if args.train_ratio is None or args.val_ratio is None:
            parser.error('train_ratio and val_ratio must be provided together')
        ratios = [args.train_ratio, args.val_ratio,
                  0.0 if args.test_ratio is None else args.test_ratio]
        if any(value < 0 or value > 1 for value in ratios):
            parser.error('split ratios must be within [0, 1]')
        if abs(sum(ratios) - 1.0) > 1e-6:
            parser.error('train_ratio + val_ratio + test_ratio must equal 1')

    modes = [args.do_distill, args.do_local_train, args.do_reverse_distill]
    if sum(bool(mode) for mode in modes) > 1:
        parser.error('distill, local-train and reverse-distill are mutually exclusive')
    if args.do_distill and not args.teacher_model_path:
        parser.error('--do_distill requires --teacher_model_path')
    if args.do_local_train and not args.pretrained_model_path:
        parser.error('--do_local_train requires --pretrained_model_path')
    if args.do_reverse_distill:
        if not args.pretrained_teacher_path or not args.student_model_path:
            parser.error('reverse distillation requires teacher and student paths')
    if args.do_distill_test and not args.do_distill:
        parser.error('--do_distill_test requires --do_distill')
    if args.do_distill_test and args.test_ratio == 0:
        parser.error('--do_distill_test requires a non-zero test split')
    if (args.use_aux_kd or args.use_ofa_kd) and not (
            args.do_distill or args.do_reverse_distill):
        parser.error('auxiliary/OFA KD flags require a distillation mode')

    split_mode = (args.do_distill or args.do_reverse_distill)
    if split_mode:
        if args.do_distill and args.model != 'CNN':
            parser.error('forward distillation requires the CNN edge student')
        if args.features != 'M':
            parser.error('cloud-edge distillation requires --features M')
        if not (args.enc_in == args.dec_in == args.c_out):
            parser.error('distillation requires enc_in == dec_in == c_out')
        if args.enc_in <= 1:
            parser.error('split distillation requires enc_in > 1')
        if (args.cloud_dim - 1) % (args.enc_in - 1) != 0:
            parser.error('(cloud_dim - 1) must be divisible by (enc_in - 1)')
    if args.model == 'TimesNet' and args.c_out != args.enc_in:
        parser.error('TimesNet requires --c_out to equal --enc_in')

    if args.use_multi_gpu:
        try:
            device_ids = [int(item) for item in args.devices.replace(' ', '').split(',')]
        except ValueError as exc:
            parser.error(f'invalid --devices: {exc}')
        if not device_ids or any(item < 0 for item in device_ids):
            parser.error('--devices must contain non-negative CUDA ids')
        args.device_ids = device_ids
        args.gpu = device_ids[0]

    cuda_available = torch.cuda.is_available()
    mps_available = (hasattr(torch.backends, 'mps')
                     and torch.backends.mps.is_available())
    available = cuda_available if args.gpu_type == 'cuda' else mps_available
    if args.use_gpu and not available:
        print(f'{args.gpu_type.upper()} unavailable; falling back to CPU.')
        args.use_gpu = 0
        args.use_multi_gpu = False
    if args.use_amp and not (args.use_gpu and args.gpu_type == 'cuda'):
        print('AMP is CUDA-only in this implementation; disabling it.')
        args.use_amp = False


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_setting(args, iteration):
    return (
        f'{args.model_id}_{args.model}_sl{args.seq_len}_pl{args.pred_len}_'
        f'dm{args.d_model}_el{args.e_layers}_{args.des}_{iteration}')


def choose_experiment(args):
    if args.do_distill:
        return Exp_OFA_Distill if args.use_ofa_kd else Exp_Distill
    if args.do_local_train:
        return Exp_LocalTrain
    if args.do_reverse_distill:
        return Exp_ReverseDistill
    return Exp_Long_Term_Forecast


def clear_cache(args):
    if args.use_gpu and args.gpu_type == 'cuda' and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif (args.use_gpu and args.gpu_type == 'mps'
          and hasattr(torch, 'mps')):
        torch.mps.empty_cache()


def main():
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    seed_everything(args.seed)
    args.device = torch.device(
        f'cuda:{args.gpu}' if args.use_gpu and args.gpu_type == 'cuda'
        else 'mps' if args.use_gpu and args.gpu_type == 'mps'
        else 'cpu')

    print('Experiment arguments:')
    for name, value in sorted(vars(args).items()):
        print(f'  {name}: {value}')

    experiment_class = choose_experiment(args)
    for iteration in range(args.itr if args.is_training else 1):
        setting = make_setting(args, iteration)
        experiment = experiment_class(args)
        if not args.is_training:
            experiment.test(setting, test=1)
        elif args.do_distill:
            experiment.distill(setting)
        elif args.do_local_train:
            experiment.local_train(setting)
        elif args.do_reverse_distill:
            experiment.update_cloud_teacher(setting)
        else:
            experiment.train(setting)
            experiment.test(setting)
        clear_cache(args)


if __name__ == '__main__':
    main()
