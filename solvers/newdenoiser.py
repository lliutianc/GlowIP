import numpy as np
import torch
import torch.nn as nn

from torchvision import datasets
import torchvision.transforms as transforms
from skimage.measure import compare_psnr, compare_ssim
import skimage.io as sio

import json
import os
import traceback
import warnings
warnings.filterwarnings("ignore")


from glow.glow import Glow
from dcgan.dcgan import Generator
from measurement.noiser import NoisyMeasurement
from measurement.noiser import *
from measurement.noiser import image_noise
from utils import gettime

transforms.ColorJitter
def solveDenoising(args):
    if args.prior == 'glow':
        GlowDenoiser(args)
    elif args.prior == 'dcgan':
        raise NotImplementedError()
    else:
        raise ValueError(f"Unrecognized `prior`: {args.prior}")


def Noiser(args, configs):
    if args.noise == 'poisson':
        # Possion noise is not additive so handle it separately.
        noise = poisson_noise(args.noise_loc, args.noise_scale)
        return noise

    elif args.noise == 'logistic':
        noise = logistic_noise(args.noise_loc, args.noise_scale)

    elif args.noise == 'gaussian':
        noise = gaussian_noise(args.noise_loc, args.noise_scale)

    elif args.noise == 'gamma':
        # raise NotImplementedError('Don\'t know how to handle gamma noise yet')
        noise = gamma_noise(args.noise_loc, args.noise_scale)

    elif args.noise == 'loggamma':
        noise = loggamma_noise(args.noise_loc, args.noise_scale)

    elif args.noise in ['glow', 'dcgan']:
        noise = image_noise(args.noise_loc, args.noise_scale,
                            noise=args.noise, size=args.size, bsz=args.batchsize,
                            configs=configs, dataset=args.dataset,
                            device=args.device)

    else:
        raise ValueError(f'Unrecognized noise distribution: {args.noise}')

    noiser = NoisyMeasurement(noise, args.noise_channel, args.noise_area, args.device)

    return noiser


def invertible_color_jitter(deviation, *augmentations):
    aug = {'brightness': (0., 0.),
           'contrast': (0., 0.),
           'saturation': (0., 0.),
           'hue': (0., 0.)}
    invert_aug = {'brightness': (0., 0.),
                  'contrast': (0., 0.),
                  'saturation': (0., 0.),
                  'hue': (0., 0.)}

    for augment in augmentations:
        if augment == 'hue':
            hue_deviation = np.clip(deviation, -.5, .5)
            aug[augment] = (hue_deviation, hue_deviation)
            invert_aug[augment] = (-hue_deviation, -hue_deviation)
        else:
            classical_deviation = 1 + deviation
            aug[augment] = (classical_deviation, classical_deviation)
            invert_aug[augment] = (1 / classical_deviation, 1 / classical_deviation)

    augmentation = torch.nn.Sequential(
        transforms.ColorJitter(brightness=aug['brightness'],
                               contrast=aug['contrast'],
                               saturation=aug['saturation'],
                               hue=aug['hue']))

    invert_augmentation = torch.nn.Sequential(
        transforms.ColorJitter(brightness=invert_aug['brightness'],
                               contrast=invert_aug['contrast'],
                               saturation=invert_aug['saturation'],
                               hue=invert_aug['hue']))

    return torch.jit.script(augmentation), torch.jit.script(invert_augmentation)


def recon_loss(noise, loc, scale):
    if noise == 'gaussian':
        def _recon(x_gen, x_noisy):
            delta = x_noisy - x_gen - loc
            delta_flat = delta.view(len(x_noisy), -1)
            nll = delta_flat ** 2
            nll_loss = nll.sum(dim=1).mean()
            return nll_loss

    elif noise == 'loggamma':
        def _recon(x_gen, x_noisy):
            delta = x_noisy - x_gen
            delta_exp = torch.exp(delta)
            nll = delta_exp / (scale - 1) - (loc - 1) * delta
            return nll.view(len(x_noisy), -1).sum(dim=1).mean()

    elif noise == 'gamma':
        def _recon(x_gen, x_noisy):
            delta = x_noisy - x_gen
            nll = delta / (scale - 1) - (loc - 1) * torch.log(delta + 1e-10)
            return nll.view(len(x_noisy), -1).sum(dim=1).mean()

    elif noise == 'poisson':
        def _recon(x_gen, x_noisy):
            noisy256 = x_noisy * 255
            gen256 = x_gen * 255

            mask = (gen256 > 0).detach().clone().float()
            mask.requires_grad = False
            valid_gen256 = gen256 * mask
            nll = valid_gen256 - noisy256 * torch.log(valid_gen256 + 1e-10)
            nll -= gen256 * (1 - mask) / 255
            return nll.view(len(x_noisy), -1).sum(dim=1).mean()

    elif noise == 'logistic':
        def _recon(x_gen, x_noisy):
            delta = x_noisy - x_gen
            z = (delta - loc) / scale
            z = z.view(len(delta), -1)
            nll = z + 2 * torch.log(1 + torch.exp(-z))
            nll = nll.sum(dim=1).mean()
            return nll

    else:
        raise ValueError(f'Unrecognized noise distribution: {noise}')

    return _recon


def GlowDenoiser(args):
    loopOver = zip(args.gamma)

    # try different gamma values
    for gamma in loopOver:
        skip_to_next  = False # flag to skip to next loop if recovery is fails due to instability
        n = args.size * args.size * 3
        # modeldir      = f"./trained_models/{args.dataset}/glow-denoising"
        modeldir = f"./trained_models/{args.dataset}/glow-cs-{args.size}"
        test_folder = f"./test_images/{args.dataset}_N=12"
        save_path = f"./results/{args.dataset}/{args.experiment}"
        # loading dataset
        trans = transforms.Compose([transforms.Resize((args.size, args.size)),
                                    transforms.ToTensor()])
        test_dataset = datasets.ImageFolder(test_folder, transform=trans)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batchsize,
                                                      drop_last=False, shuffle=False)
        # loading glow configurations
        config_path = modeldir+"/configs.json"
        with open(config_path, 'r') as f:
            configs = json.load(f)

        # noiser
        noiser = Noiser(args, configs)
        # loss
        loss = recon_loss(args.noise, args.noise_loc, args.noise_scale)
        # regularizor
        gamma = torch.tensor(gamma, requires_grad=False, dtype=torch.float, device=args.device)
        # augmentations
        if args.augmentation:
            aug, inv_aug = invertible_color_jitter(args.augmentation_deviation, args.augmentation)
        else:
            aug, inv_aug = None, None
        # results to save
        Original = []
        Noisy = []
        Noise = []

        Original_base = []
        Noisy_base = []

        Recovered = []
        Recovered_per_10_steps = {}
        Recovered_base_per_10_steps = {}

        Residual_Curve = []
        PSNR_Curve = []
        gamma_Curve = []
        for i, data in enumerate(test_dataloader):

            x_test = data[0]
            x_test = x_test.clone().to(device=args.device)
            if aug:
                x_test = aug(x_test)
            n_test = x_test.size()[0]
            assert n_test == args.batchsize, \
                "please make sure that no. of images are evenly divided by batchsize"

            # loading glow model
            glow = Glow((3,args.size,args.size),
                        K=configs["K"],L=configs["L"],
                        coupling=configs["coupling"],
                        n_bits_x=configs["n_bits_x"],
                        nn_init_last_zeros=configs["last_zeros"],
                        device=args.device)
            glow.load_state_dict(torch.load(modeldir+"/glowmodel.pt", map_location=args.device))
            glow.eval()

            # add noise
            noise = noiser(x_test)
            x_noisy = x_test + noise
            x_noisy = torch.clamp(x_noisy, 0., 1.)

            # making a forward to record shapes of z's for reverse pass

            _ = glow(glow.preprocess(torch.zeros_like(x_test)))

            # np.random.seed(args.random_seed)
            if args.init_strategy == "random":
                z_sampled = np.random.normal(0, args.init_std, [n_test, n])

            elif args.init_strategy == "from-noisy":
                z, _, _ = glow(glow.preprocess(x_noisy * 255, clone=True))
                z = glow.flatten_z(z)
                z_sampled = z.clone().detach().cpu().numpy()

            elif args.init_strategy == 'from-real':
                z, _, _ = glow(glow.preprocess(x_test * 255, clone=True))
                z = glow.flatten_z(z)
                z_sampled = z.clone().detach().cpu().numpy()

            else:
                raise ValueError("Unrecognized initialization strategy")

            z_sampled = torch.from_numpy(z_sampled).float().to(args.device)
            z_sampled = nn.Parameter(z_sampled, requires_grad=True)

            # optimizer

            if args.optim == "adam":
                optimizer = torch.optim.Adam([z_sampled], lr=args.lr,)
            elif args.optim == "lbfgs":
                optimizer = torch.optim.LBFGS([z_sampled], lr=args.lr,)

            # to be recorded over iteration
            z_original_unflat = glow(glow.preprocess(x_test * 255, clone=True))[0]
            z_original_np = glow.flatten_z(z_original_unflat).data.cpu().numpy()
            Original_base.append(z_original_np)

            z_noisy_unflat = glow(glow.preprocess(x_noisy * 255, clone=True))[0]
            z_noisy_np = glow.flatten_z(z_noisy_unflat).data.cpu().numpy()
            Noisy_base.append(z_noisy_np)

            psnr_t = torch.nn.MSELoss().to(device=args.device)
            residual = []
            psnr_hist = []
            gamma_hist = []

            # save initial results.

            t = 0
            z_unflat = glow.unflatten_z(z_sampled, clone=True)
            x_gen = glow(z_unflat, reverse=True, reverse_clone=True)
            x_gen = glow.postprocess(x_gen, floor_clamp=False)
            if inv_aug:
                x_gen = inv_aug(x_gen)

            x_gen_np = x_gen.data.cpu().numpy().transpose(0, 2, 3, 1)
            x_gen_np = np.clip(x_gen_np, 0, 1)
            if t in Recovered_per_10_steps:
                Recovered_per_10_steps[t].append(x_gen_np)
            else:
                Recovered_per_10_steps[t] = [x_gen_np]

            z_flatted = z_sampled.data.cpu().numpy()
            if t in Recovered_base_per_10_steps:
                Recovered_base_per_10_steps[t].append(z_flatted)
            else:
                Recovered_base_per_10_steps[t] = [z_flatted]
            del z_unflat, x_gen

            # Optimizer
            if args.train_strategy == 'restart':
                pass

            else:
                eval_psnr = None
                for t in range(1, args.steps + 1):
                    try:
                        def closure():
                            global residual_t, psnr

                            optimizer.zero_grad()
                            z_unflat = glow.unflatten_z(z_sampled, clone=False)
                            x_gen = glow(z_unflat, reverse=True, reverse_clone=False)
                            x_gen = glow.postprocess(x_gen, floor_clamp=False)

                            residual_t = loss(x_gen, x_noisy)
                            if not args.z_penalty_unsquared:
                                z_reg_loss_t = gamma * (z_sampled.norm(dim=1)**2).mean()
                            else:
                                z_reg_loss_t = gamma * z_sampled.norm(dim=1).mean()
                            loss_t = residual_t + z_reg_loss_t
                            loss_t.backward(retain_graph=True)

                            psnr = psnr_t(x_noisy, x_gen)
                            psnr = 10 * np.log10(1 / psnr.item())
                            print(f'\rStep={t}|'
                                  f'Loss={loss_t.item():.4f}|'
                                  f'Residual={residual_t.item():.4f}|'
                                  f'Reg_z={z_reg_loss_t.item():.5f}|gamma={gamma.item():.4f}|'
                                  f'PSNR(noisy)={psnr:.3f}', end='\r')

                            return loss_t

                        optimizer.step(closure)
                        residual.append(residual_t.item())
                        psnr_hist.append(psnr)
                        gamma_hist.append(gamma.item())

                        if t % args.eval_every == 0:
                            z_unflat = glow.unflatten_z(z_sampled, clone=True)
                            x_gen = glow(z_unflat, reverse=True, reverse_clone=True)
                            x_gen = glow.postprocess(x_gen, floor_clamp=False)
                            if inv_aug:
                                x_gen = inv_aug(x_gen)

                            x_gen_np = x_gen.data.cpu().numpy().transpose(0, 2, 3, 1)
                            x_gen_np = np.clip(x_gen_np, 0, 1)
                            if t in Recovered_per_10_steps:
                                Recovered_per_10_steps[t].append(x_gen_np)
                            else:
                                Recovered_per_10_steps[t] = [x_gen_np]

                            z_flatted = z_sampled.data.cpu().numpy()
                            if t in Recovered_base_per_10_steps:
                                Recovered_base_per_10_steps[t].append(z_flatted)
                            else:
                                Recovered_base_per_10_steps[t] = [z_flatted]

                            if args.train_strategy == 'bilevel':
                                # Evaluate if too much noise was modeled by GLOW.
                                # If so, we increase gamma to force base to move towards 0.
                                # Else, we decrease gamma.

                                base_step = 0.5
                                increase_step = 0

                                delta = x_noisy - x_gen
                                delta = delta.view(len(x_noisy), -1).data.cpu().numpy()

                                if eval_psnr is not None:
                                    # If PSNR between recovered and noisy was increased,
                                    # then too much noise was modeled by GLOW.
                                    increase_step += psnr - eval_psnr
                                eval_psnr = psnr

                                if args.noise == 'gaussian':
                                    # If the std of gaussian noise is smaller than the known scale,
                                    # then too much noise was modeled by GLOW.
                                    increase_step += (args.noise_scale - delta.std(1)).sum()

                                print(increase_step, '\n')
                                gamma *= (1 + base_step * increase_step)

                            del z_unflat, x_gen, x_gen_np, z_flatted

                    except Exception as e:
                        traceback.print_exc()
                        skip_to_next = True
                        break

            # if skip_to_next:
            #     break

            x_test_np = x_test.data.cpu().numpy().transpose(0, 2, 3, 1)
            Original.append(x_test_np)

            noise_np = noise.data.cpu().numpy().transpose(0, 2, 3, 1)
            Noise.append(noise_np)

            x_noisy_np = x_noisy.data.cpu().numpy().transpose(0, 2, 3, 1)
            Noisy.append(x_noisy_np)

            Residual_Curve.append(residual)
            PSNR_Curve.append(psnr_hist)
            gamma_Curve.append(gamma_hist)

            x_gen = None
            try:
                z_unflat = glow.unflatten_z(z_sampled, clone=False)
                x_gen = glow(z_unflat, reverse=True, reverse_clone=False)
                x_gen = glow.postprocess(x_gen, floor_clamp=False)
                if inv_aug:
                    x_gen = inv_aug(x_gen)

                x_gen_np = x_gen.data.cpu().numpy().transpose(0, 2, 3, 1)
                x_gen_np = np.clip(x_gen_np, 0, 1)
                Recovered.append(x_gen_np)
            except Exception as e:
                traceback.print_exc()

            # freeing up memory for second loop
            glow.zero_grad()
            optimizer.zero_grad()
            del x_test, x_gen, optimizer, psnr_t, z_sampled, glow, noise
            with torch.cuda.device(args.device):
                torch.cuda.empty_cache()
            print("\nbatch completed")

            # todo: remove this break after finishing development.
            break

        # if skip_to_next:
        #     print("\nskipping current loop due to instability or user triggered quit")
        #     continue

        # metric evaluations

        Original = np.vstack(Original)
        Noisy = np.vstack(Noisy)
        Noise = np.vstack(Noise)
        Original_base = np.vstack(Original_base)
        Noisy_base = np.vstack(Noisy_base)

        # print performance analysis
        printout = "+-"*10 + "%s"%args.dataset + "-+"*10 + "\n"
        printout = printout + "\t n_test = %d\n"%len(Recovered)
        printout = printout + "\t noise_std = %0.4f\n"%args.noise_scale
        printout = printout + "\t gamma = %0.6f\n"%gamma
        print(printout)

        if args.save_metrics_text:
            with open("%s_denoising_glow_results.txt"%args.dataset,"a") as f:
                f.write('\n' + printout)

        # saving images

        if args.save_results:
            gamma = gamma.item()
            file_names = [name[0].split("/")[-1].split(".")[0] for name in test_dataset.samples]

            save_path = os.path.join(save_path, f'{args.noise}_'
                                                f'{args.noise_loc}#{args.noise_scale}_'
                                                f'{args.noise_channel}_{args.noise_area}_'
                                                f'{args.init_strategy}_'
                                                f'{round(gamma, 4)}_{gettime()}')

            if not os.path.exists(save_path):
                os.makedirs(save_path)
            else:
                save_path_1 = save_path + "_1"
                if not os.path.exists(save_path_1):
                    os.makedirs(save_path_1)
                    save_path = save_path_1
                else:
                    save_path_2 = save_path + "_2"
                    if not os.path.exists(save_path_2):
                        os.makedirs(save_path_2)
                        save_path = save_path_2

            Residual_Curve = np.array(Residual_Curve).mean(axis=0)
            PSNR_Curve = np.array(PSNR_Curve).mean(axis=0)
            gamma_Curve = np.array(gamma_Curve).mean(axis=0)
            np.save(save_path + "/residual_curve.npy", Residual_Curve)
            np.save(save_path + "/psnr_curve.npy", PSNR_Curve)
            np.save(save_path + "/gamma_curve.npy", gamma_Curve)

            np.save(save_path + "/original.npy", Original)
            np.save(save_path + "/noisy.npy", Noisy)
            np.save(save_path + "/noise.npy", Noise)

            np.save(save_path + "/base_noisy.npy", Noisy_base)
            np.save(save_path + "/base_original.npy", Original_base)

            if len(Recovered_per_10_steps) > 0:
                for t, checkpoints in Recovered_per_10_steps.items():
                    try:
                        np.save(save_path+f'/recovered_{t}.npy', np.vstack(checkpoints))
                    except Exception as e:
                        traceback.print_exc()
                        break

            if len(Recovered_base_per_10_steps) > 0:
                for t, checkpoints in Recovered_base_per_10_steps.items():
                    try:
                        np.save(save_path+f'/base_{t}.npy', np.vstack(checkpoints))
                    except Exception as e:
                        traceback.print_exc()
                        break


