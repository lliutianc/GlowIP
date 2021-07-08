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


def householder_caster(b, n, device):
    I = torch.eye(n, device=device, requires_grad=False)

    def compute_householder_matrix(vs):
        Qs = []
        for i in range(b):
            Q = torch.eye(n, device=device)
            for v in vs:
                vi = v[i].view(-1, 1)
                vi = vi / vi.norm()
                Qi = I - 2 * torch.mm(vi, vi.permute(1, 0))
                Q = torch.mm(Q, Qi)
            Qs.append(Q)
        return torch.stack(Qs)
    return compute_householder_matrix


def GlowDenoiser(args):
    if args.noise != 'gaussian':
        raise ValueError()

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

    noiser = Noiser(args, configs)
    loss = recon_loss(args.noise, args.noise_loc, args.noise_scale)

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
    for i, data in enumerate(test_dataloader):

        x_test = data[0]
        x_test = x_test.clone().to(device=args.device)
        n_test = x_test.size()[0]
        assert n_test == args.batchsize, \
            "please make sure that no. of images are evenly divided by batchsize"

        # loading glow model
        glow = Glow((3, args.size, args.size),
                    K=configs["K"], L=configs["L"],
                    coupling=configs["coupling"],
                    n_bits_x=configs["n_bits_x"],
                    nn_init_last_zeros=configs["last_zeros"],
                    device=args.device)
        glow.load_state_dict(torch.load(modeldir+"/glowmodel.pt", map_location=args.device))
        glow.eval()

        householder = householder_caster(n_test, n, args.device)

        # add noise
        noise = noiser(x_test)
        x_noisy = x_test + noise
        x_noisy = torch.clamp(x_noisy, 0., 1.)

        # making a forward to record shapes of z's for reverse pass
        _ = glow(glow.preprocess(torch.zeros_like(x_test)))

        noise_estimate = np.random.normal(args.noise_loc, args.noise_scale, [n_test, n, 1])
        noise_estimate = torch.from_numpy(noise_estimate).float().to(args.device)
        noise_estimate = nn.Parameter(noise_estimate, requires_grad=False)

        householder_iter = args.householder_iter or n
        vs = [nn.Parameter(torch.randn(n_test, n, device=args.device),
                           requires_grad=True) for _ in range(householder_iter)]
        # optimizer
        if args.optim == "adam":
            optimizer = torch.optim.Adam(vs, lr=args.lr,)
        elif args.optim == "lbfgs":
            optimizer = torch.optim.LBFGS(vs, lr=args.lr,)

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

        # save initial results.
        t = 0
        noise_recov = householder(vs) @ noise_estimate
        noise_recov = noise_recov.view(n_test, args.size, args.size)
        x_gen = x_noisy - noise_recov
        z = glow(glow.preprocess(x_gen * 255, clone=True))

        x_gen_np = x_gen.data.cpu().numpy().transpose(0, 2, 3, 1)
        x_gen_np = np.clip(x_gen_np, 0, 1)
        z_flat = glow.flatten_z(z).data.cpu().numpy()

        if t in Recovered_per_10_steps:
            Recovered_per_10_steps[t].append(x_gen_np)
        else:
            Recovered_per_10_steps[t] = [x_gen_np]

        if t in Recovered_base_per_10_steps:
            Recovered_base_per_10_steps[t].append(z_flat)
        else:
            Recovered_base_per_10_steps[t] = [z_flat]

        del z_flat, z, x_gen, x_gen_np, noise_recov

        for t in range(1, args.steps + 1):
            try:
                def closure():
                    global nll, psnr
                    optimizer.zero_grad()
                    noise_flat = householder(vs) @ noise_estimate
                    noise_recov = noise_flat.view(n_test, args.size, args.size)
                    x_gen = x_noisy - noise_recov
                    nll, logdet, logpz, z_mu, z_std = glow.nll_loss(glow.preprocess(x_gen))
                    nll.backward(retain_graph=True)
                    torch.nn.utils.clip_grad_value_(vs, 5)

                    psnr = psnr_t(x_noisy, x_gen)
                    psnr = 10 * np.log10(1 / psnr.item())
                    print(f'\rStep={t}|'
                          f'Loss={nll.item():.4f}|'
                          f'PSNR(noisy)={psnr:.3f}', end='\r')

                    return nll

                optimizer.step(closure)
                residual.append(nll.item())
                psnr_hist.append(psnr)

                if t % args.eval_every == 0:
                    noise_recov = householder(vs) @ noise_estimate
                    noise_recov = noise_recov.view(n_test, args.size, args.size)
                    x_gen = x_noisy - noise_recov
                    z = glow(glow.preprocess(x_gen * 255, clone=True))

                    x_gen_np = x_gen.data.cpu().numpy().transpose(0, 2, 3, 1)
                    x_gen_np = np.clip(x_gen_np, 0, 1)
                    z_flat = glow.flatten_z(z).data.cpu().numpy()

                    if t in Recovered_per_10_steps:
                        Recovered_per_10_steps[t].append(x_gen_np)
                    else:
                        Recovered_per_10_steps[t] = [x_gen_np]

                    if t in Recovered_base_per_10_steps:
                        Recovered_base_per_10_steps[t].append(z_flat)
                    else:
                        Recovered_base_per_10_steps[t] = [z_flat]

                    del z_flat, z, x_gen, x_gen_np, noise_recov

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

        x_gen = None
        try:
            noise_recov = householder(vs) @ noise_estimate
            noise_recov = noise_recov.view(n_test, args.size, args.size)
            x_gen = x_noisy - noise_recov
            x_gen_np = x_gen.data.cpu().numpy().transpose(0, 2, 3, 1)
            x_gen_np = np.clip(x_gen_np, 0, 1)
            Recovered.append(x_gen_np)
        except Exception as e:
            traceback.print_exc()

        # freeing up memory for second loop
        glow.zero_grad()
        optimizer.zero_grad()
        del x_test, x_gen, optimizer, psnr_t, noise_recov, glow, noise
        with torch.cuda.device(args.device):
            torch.cuda.empty_cache()
        print("\nbatch completed")

        # todo: remove this break after finishing development.
        break

    Original = np.vstack(Original)
    Noisy = np.vstack(Noisy)
    Noise = np.vstack(Noise)
    Original_base = np.vstack(Original_base)
    Noisy_base = np.vstack(Noisy_base)

    # print performance analysis
    printout = "+-"*10 + "%s"%args.dataset + "-+"*10 + "\n"
    printout = printout + "\t n_test = %d\n"%len(Recovered)
    printout = printout + "\t noise_std = %0.4f\n"%args.noise_scale
    print(printout)

    if args.save_metrics_text:
        with open("%s_denoising_glow_results.txt"%args.dataset,"a") as f:
            f.write('\n' + printout)

    # saving images

    if args.save_results:
        file_names = [name[0].split("/")[-1].split(".")[0] for name in test_dataset.samples]

        save_path = os.path.join(save_path, f'{args.noise}_'
                                            f'{args.noise_loc}#{args.noise_scale}_'
                                            f'{args.noise_channel}_{args.noise_area}_'
                                            f'householder_{gettime()}')

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
        np.save(save_path+"/residual_curve.npy", Residual_Curve)
        np.save(save_path+"/psnr_curve.npy", PSNR_Curve)

        np.save(save_path+"/original.npy", Original)
        np.save(save_path+"/noisy.npy", Noisy)
        np.save(save_path+"/noise.npy", Noise)

        np.save(save_path+"/base_noisy.npy", Noisy_base)
        np.save(save_path+"/base_original.npy", Original_base)

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


