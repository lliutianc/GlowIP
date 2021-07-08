import argparse
# from solvers.denoiser import solveDenoising
# from solvers.redenoiser import solveDenoising as solveDenoisingWithRestart
from solvers.newdenoiser import solveDenoising
import torch


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='solve denoising')
    # task details: prior, dataset, image size
    parser.add_argument('-experiment', type=str, help='the name of experiment', default='denoising')
    parser.add_argument('-restart_denoise', type=int, default=1)
    parser.add_argument('-train_strategy', type='str', default='bilevel', choices=['none', 'bilevel', 'restart'])
    parser.add_argument('-prior', type=str, help='choose with prior to use glow, dcgan', default='glow')
    parser.add_argument('-dataset', type=str, help='the dataset/images to use', default='celeba')
    parser.add_argument('-size', type=int, help='size of images to resize all images to',
                        default=64)
    # noise
    parser.add_argument('-noise', type=str, help='type of noise to add', default='gaussian')
    parser.add_argument('-noise_scale', type=float, help='scale of noise to add', default=0.1)
    parser.add_argument('-noise_loc', type=float, help='loc of noise to add', default=0)
    parser.add_argument('-noise_channel', type=int, help='the first C channels of noise to add', default=None)
    parser.add_argument('-noise_area', type=int, help='the area of noise to add', default=None)

    # training
    # parser.add_argument('-gamma',  type=float, nargs='+', help='regularizor', default=[0.0125])
    parser.add_argument('-gamma',  type=float, nargs='+', help='regularizor', default=[0, 0.01, 0.0125, 0.025, 0.05, 0.075, 0.1])
    parser.add_argument('-init_gamma',  type=float, default=0.05, help='regularizor in stage 1')
    # parser.add_argument('-gamma',  type=float, nargs='+', help='regularizor', default=[0, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2])
    # parser.add_argument('-gamma',  type=float, nargs='+', help='regularizor', default=[0, 0.01, 0.0125, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 20, 30, 40, 50])
    parser.add_argument('-optim', type=str, help='optimizer', default="lbfgs")
    parser.add_argument('-lr', type=float, help='learning rate', default=1)
    parser.add_argument('-steps', type=int, help='no. of steps to run', default=50)
    parser.add_argument('-eval_every', type=int, default=10)
    parser.add_argument('-batchsize', type=int, help='no. of images to solve in parallel as batches', default=6)
    parser.add_argument('-z_penalty_unsquared', action="store_true", help="use ||z|| if True else ||z||^2")

    # trainable parameters
    parser.add_argument('-init_strategy', type=str, help="init strategy to use", default='random')
    parser.add_argument('-init_std', type=float, help='std of init_strategy is random', default=0.2)

    # save and other config
    parser.add_argument('-save_metrics_text', type=bool, help='whether to save results to a text file', default=True)
    parser.add_argument('-save_results', type=bool, help='whether to save results after experiments conclude', default=True)
    parser.add_argument('-cuda', type=int, help='which gpu to use', default=6)
    parser.add_argument('-dev', action='store_true', default=False)
    parser.add_argument('-random_seed', type=int, default=2021)

    args = parser.parse_args()
    if args.dev:
        args.device = torch.device(f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu')

    else:
        # formal situation will handle the cuda allocation automatically
        args.device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')

    solveDenoising(args)
    #
    # if args.restart_denoise:
    #     solveDenoisingWithRestart(args)
    # else:
    #     solveDenoising(args)
