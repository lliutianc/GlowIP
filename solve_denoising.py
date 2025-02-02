import argparse
from solvers.newdenoiser import solveDenoising
from solvers.householder_denoiser import solveDenoising as householderDenoiser

import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='solve denoising')
    # task details: prior, dataset, image size
    parser.add_argument('-experiment', type=str, help='the name of experiment', default='denoising')
    parser.add_argument('-restart_denoise', type=int, default=1)
    parser.add_argument('-train_strategy', type=str, default='bilevel',
                        choices=['none', 'bilevel', 'restart', 'householder'])
    parser.add_argument('-augmentation', type=str, nargs='+', default=[])
    parser.add_argument('-householder_iter', type=int, default=10)

    parser.add_argument('-prior', type=str, help='choose with prior to use glow, dcgan', default='glow')
    parser.add_argument('-dataset', type=str, help='the dataset/images to use', default='celeba')
    parser.add_argument('-size', type=int, help='size of images to resize all images to',
                        default=64)
    # noise
    parser.add_argument('-noise', type=str, help='type of noise to add', default='gaussian')
    parser.add_argument('-noise_scale', type=float, help='scale of noise to add', default=0.1)
    parser.add_argument('-noise_loc', type=float, help='loc of noise to add', default=0)
    parser.add_argument('-noise_channel', type=int, default=None,
                        help='the first C channels of noise to add')
    parser.add_argument('-noise_area', type=int, help='the area of noise to add', default=None)

    # training
    parser.add_argument('-augmentation_deviation', type=float, default=.2,
                        help='deviation in invertible augmentations', )
    parser.add_argument('-gamma',  type=float, nargs='+', help='regularizor',
                        default=[0, 0.01, 0.0125, 0.025, 0.05, 0.075, 0.1], )
    parser.add_argument('-init_gamma',  type=float, default=0.05, help='regularizor in stage 1')
    parser.add_argument('-optim', type=str, help='optimizer', default="lbfgs")
    parser.add_argument('-lr', type=float, help='learning rate', default=1)
    parser.add_argument('-steps', type=int, help='no. of steps to run', default=5)
    parser.add_argument('-eval_every', type=int, default=10)
    parser.add_argument('-batchsize', type=int, default=6)
    parser.add_argument('-z_penalty_unsquared', action="store_true",
                        help="use ||z|| if True else ||z||^2")

    # trainable parameters
    parser.add_argument('-init_strategy', type=str, help="init strategy to use", default='random')
    parser.add_argument('-init_std', type=float, help='std of init_strategy is random', default=0.2)

    # save and other config
    parser.add_argument('-save_metrics_text', type=bool, default=True,
                        help='whether to save results to a text file')
    parser.add_argument('-save_results', type=bool, default=True,
                        help='whether to save results after experiments conclude')
    parser.add_argument('-cuda', type=int, help='which gpu to use', default=6)
    parser.add_argument('-dev', action='store_true', default=False)
    parser.add_argument('-random_seed', type=int, default=2021)

    args = parser.parse_args()

    if args.dev:
        args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    else:
        # formal situation will handle the cuda allocation automatically
        args.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    if len(args.augmentation) >= 1:
        # augmentations = ['brightness', 'contrast', 'saturation', 'hue']
        augmentations = args.augmentation
        for aug in augmentations:
            args.augmentation = [aug]
            if args.train_strategy == 'householder':
                householderDenoiser(args)
            else:
                solveDenoising(args)
    else:
        if args.train_strategy == 'householder':
            householderDenoiser(args)
        else:
            solveDenoising(args)
