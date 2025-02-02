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
        GANDenoiser(args)
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

        # def _recon(x_gen, x_noisy):
        #     delta = x_noisy - x_gen - loc
        #     delta_flat = delta.view(len(x_noisy), -1)
        #     # nll = delta_flat ** 2
        #     nll_loss = delta_flat.mean(1) ** 2
        #     nll_loss = nll_loss.mean()
        #     # nll_loss = nll.sum(dim=1).mean()
        #
        #     scale_loss = delta_flat.std(dim=1)
        #     scale_loss = (scale_loss - scale) ** 2
        #     # print(scale_loss.mean().item(), nll_loss.mean().item())
        #     nll_loss += scale_loss.mean()
        #
        #     return nll_loss * 100.

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
            # nll_term = nll.view(len(x_noisy), -1).sum(dim=1).mean()
            # penalty_term = (delta * (1 - mask)).sum()
            #
            # print(nll_term, penalty_term)
            # return nll_term + penalty_term
            #

            # nll = gen256 - noisy256 * torch.log(gen256 + 1e-10)
            # print(torch.log(gen256 + 1e-10).mean().item(), gen256.min().item(), '\n')
            return nll.view(len(x_noisy), -1).sum(dim=1).mean()

    elif noise == 'logistic':
        def _recon(x_gen, x_noisy):
            delta = x_noisy - x_gen
            z = (delta - loc) / scale
            z = z.view(len(delta), -1)

            # z_min_per_obs = torch.min(z, dim=1, keepdim=True)[0]
            # nll1 = z - z_min_per_obs
            # nll2 = torch.log(torch.exp(z_min_per_obs - z) + torch.exp(z_min_per_obs) + 1e-10)
            # nll = nll1 + 2 * nll2
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
        n             = args.size*args.size*3
        # modeldir      = f"./trained_models/{args.dataset}/glow-denoising"
        modeldir      = f"./trained_models/{args.dataset}/glow-cs-{args.size}"
        test_folder   = f"./test_images/{args.dataset}_N=12"
        save_path     = f"./results/{args.dataset}/{args.experiment}"

        # loading dataset
        trans           = transforms.Compose([transforms.Resize((args.size,args.size)),transforms.ToTensor()])
        test_dataset    = datasets.ImageFolder(test_folder, transform=trans)
        test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batchsize,drop_last=False,shuffle=False)

        # loading glow configurations
        config_path = modeldir+"/configs.json"
        with open(config_path, 'r') as f:
            configs = json.load(f)

        # noiser
        noiser = Noiser(args, configs)

        # loss
        loss = recon_loss(args.noise, args.noise_loc, args.noise_scale)

        # regularizor
        gamma = torch.tensor(gamma, requires_grad=True, dtype=torch.float, device=args.device)

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

            # selecting optimizer

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

            # save initial results.
            t = 0
            z_unflat = glow.unflatten_z(z_sampled, clone=True)
            x_gen = glow(z_unflat, reverse=True, reverse_clone=True)
            x_gen = glow.postprocess(x_gen, floor_clamp=False)

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

            for t in range(1, args.steps+1):
                try:
                    def closure():
                        optimizer.zero_grad()
                        z_unflat = glow.unflatten_z(z_sampled, clone=False)
                        x_gen = glow(z_unflat, reverse=True, reverse_clone=False)
                        x_gen = glow.postprocess(x_gen, floor_clamp=False)
                        global residual_t
                        residual_t = loss(x_gen, x_noisy)
                        if not args.z_penalty_unsquared:
                            z_reg_loss_t = gamma * (z_sampled.norm(dim=1)**2).mean()
                        else:
                            z_reg_loss_t = gamma * z_sampled.norm(dim=1).mean()
                        loss_t = residual_t + z_reg_loss_t
                        global psnr
                        psnr = psnr_t(x_test, x_gen)
                        psnr = 10 * np.log10(1 / psnr.item())
                        print("\rAt step=%0.3d|loss=%0.4f|residual=%0.4f|z_reg=%0.5f|psnr=%0.3f"%(
                            t, loss_t.item(), residual_t.item(), z_reg_loss_t.item(), psnr),
                              end="\r")
                        loss_t.backward(retain_graph=True)
                        return loss_t
                    optimizer.step(closure)
                    residual.append(residual_t.item())
                    psnr_hist.append(psnr)

                    if t % 10 == 0:
                        z_unflat = glow.unflatten_z(z_sampled, clone=True)
                        x_gen = glow(z_unflat, reverse=True, reverse_clone=True)
                        x_gen = glow.postprocess(x_gen, floor_clamp=False)

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
                z_unflat = glow.unflatten_z(z_sampled, clone=False)
                x_gen = glow(z_unflat, reverse=True, reverse_clone=False)
                x_gen = glow.postprocess(x_gen, floor_clamp=False)

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

        # # Save final results
        # psnr = None
        # try:
        #     Recovered = np.vstack(Recovered)
        #     psnr = [compare_psnr(x, y) for x,y in zip(Original, Recovered)]
        # except Exception as e:
        #     continue

        # print performance analysis

        printout = "+-"*10 + "%s"%args.dataset + "-+"*10 + "\n"
        printout = printout + "\t n_test = %d\n"%len(Recovered)
        printout = printout + "\t noise_std = %0.4f\n"%args.noise_scale
        printout = printout + "\t gamma = %0.6f\n"%gamma
        # if psnr is not None:
        #     printout = printout + "\t PSNR = %0.3f\n"%np.mean(psnr)
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
            # _ = [sio.imsave(save_path+"/"+name+"_noisy.jpg", x) for x, name in zip(Noisy, file_names)]

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

            # if len(Recovered) > 0:
            #     np.save(save_path + "/recovered_final.npy", Recovered)
            #     _ = [sio.imsave(save_path + "/" + name + "_recov.jpg", x) for x, name in
            #          zip(Recovered, file_names)]


def GANDenoiser(args):
    raise NotImplementedError('Haven\'t supported GAN yet')

    # assert args.noise == "gaussian", "only Gaussian noise is supported in GANDenoiser"
    # loopOver = zip(args.gamma)
    # for gamma in loopOver:
    #     n             = 100
    #     modeldir      = "./trained_models/%s/dcgan"%args.dataset
    #     test_folder   = "./test_images/%s"%args.dataset
    #     save_path     = "./results/%s/%s"%(args.dataset,args.experiment)
    #     # loading dataset
    #     trans           = transforms.Compose([transforms.Resize((args.size,args.size)),transforms.ToTensor()])
    #     test_dataset    = datasets.ImageFolder(test_folder, transform=trans)
    #     test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=args.batchsize,drop_last=False,shuffle=False)
    #     # regularizor
    #     gamma     = torch.tensor(gamma, requires_grad=True, dtype=torch.float, device=args.device)
    #     # getting test images
    #     Original  = []
    #     Recovered = []
    #     Noisy     = []
    #     Residual_Curve = []
    #     for i, data in enumerate(test_dataloader):
    #         # getting batch of data
    #         x_test = data[0]
    #         x_test = x_test.clone().to(device=args.device)
    #         n_test = x_test.size()[0]
    #         assert n_test == args.batchsize, "please make sure that no. of images are evenly divided by batchsize"
    #
    #         # noise to be added
    #         noise = np.random.normal(0,args.noise_std,size=(n_test,3,args.size,args.size))
    #         noise = torch.tensor(noise,dtype=torch.float,requires_grad=False, device=args.device)
    #
    #         # loading dcgan model
    #         generator = Generator(ngpu=1).to(device=args.device)
    #         generator.load_state_dict(torch.load(modeldir+'/dcgan_G.pt'))
    #         generator.eval()
    #
    #         # initializing z's
    #         z_sampled = np.random.normal(0,args.init_std,[n_test,n,1,1])
    #         z_sampled = torch.tensor(z_sampled,requires_grad=True,dtype=torch.float,device=args.device)
    #
    #         # selecting optimizer
    #         if args.optim == "adam":
    #             optimizer = torch.optim.Adam([z_sampled], lr=args.lr,)
    #         elif args.optim == "lbfgs":
    #             optimizer = torch.optim.LBFGS([z_sampled], lr=args.lr,)
    #
    #         # metrics to record over training
    #         psnr_t    = torch.nn.MSELoss().to(device=args.device)
    #         residual  = []
    #
    #         # running optimizer steps
    #         for t in range(args.steps):
    #             def closure():
    #                 optimizer.zero_grad()
    #                 x_gen       = generator(z_sampled)
    #                 x_gen       = (x_gen + 1)/2
    #                 x_noisy     = x_test + noise
    #                 global residual_t
    #                 residual_t  = ((x_gen - x_noisy)**2).view(len(x_noisy),-1).sum(dim=1).mean()
    #                 if args.z_penalty_unsquared:
    #                     z_reg_loss_t= gamma*z_sampled.norm(dim=1).mean()
    #                 else:
    #                     z_reg_loss_t= gamma*(z_sampled.norm(dim=1)**2).mean()
    #                 loss_t      = residual_t + z_reg_loss_t
    #                 psnr = torch.square(x_test - x_gen).mean()
    #                 psnr = 10 * np.log10(1 / psnr.item())
    #                 # psnr        = psnr_t(x_test, x_gen)
    #                 print("\rAt step=%0.3d|loss=%0.4f|residual=%0.4f|z_reg=%0.5f|psnr=%0.3f"%(t,loss_t.item(),residual_t.item(),z_reg_loss_t.item(), psnr),end="\r")
    #                 loss_t.backward()
    #                 return loss_t
    #             optimizer.step(closure)
    #             residual.append(residual_t.item())
    #
    #         # getting recovered and true images
    #         x_test_np  = x_test.data.cpu().numpy().transpose(0,2,3,1)
    #         x_gen      = generator(z_sampled)
    #         x_gen      = (x_gen + 1)/2
    #         x_gen_np   = x_gen.data.cpu().numpy().transpose(0,2,3,1)
    #         x_gen_np   = np.clip(x_gen_np,0,1)
    #         x_noisy    = x_test + noise
    #         x_noisy_np = x_noisy.data.cpu().numpy().transpose(0,2,3,1)
    #         x_noisy_np = np.clip(x_noisy_np,0,1)
    #
    #         Original.append(x_test_np)
    #         Recovered.append(x_gen_np)
    #         Noisy.append(x_noisy_np)
    #         Residual_Curve.append(residual)
    #
    #         # freeing up memory for second loop
    #         generator.zero_grad()
    #         optimizer.zero_grad()
    #         del x_test, x_gen, optimizer, psnr_t, z_sampled, generator, noise,
    #         torch.cuda.empty_cache()
    #         print("\nbatch completed")
    #
    #     # metric evaluations
    #     Original  = np.vstack(Original)
    #     Recovered = np.vstack(Recovered)
    #     Noisy     = np.vstack(Noisy)
    #     psnr      = [compare_psnr(x, y) for x,y in zip(Original, Recovered)]
    #
    #     # print performance analysis
    #     printout = "+-"*10 + "%s"%args.dataset + "-+"*10 + "\n"
    #     printout = printout + "\t n_test     = %d\n"%len(Recovered)
    #     printout = printout + "\t noises_std = %0.4f\n"%args.noise_std
    #     printout = printout + "\t gamma      = %0.6f\n"%gamma
    #     printout = printout + "\t PSNR       = %0.3f\n"%np.mean(psnr)
    #     print(printout)
    #     if args.save_metrics_text:
    #         with open("%s_denoising_dcgan_results.txt"%args.dataset,"a") as f:
    #             f.write('\n' + printout)
    #
    #     # saving images
    #     if args.save_results:
    #         gamma = gamma.item()
    #         file_names = [name[0].split("/")[-1].split(".")[0] for name in test_dataset.samples]
    #         save_path = save_path + "/denoising_noisestd_%0.4f_gamma_%0.6f_steps_%d_lr_%0.3f_init_std_%0.2f_optim_%s"
    #         save_path = save_path%(args.noise_std, gamma, args.steps, args.lr, args.init_std, args.optim)
    #         if not os.path.exists(save_path):
    #             os.makedirs(save_path)
    #         else:
    #             save_path_1 = save_path + "_1"
    #             if not os.path.exists(save_path_1):
    #                 os.makedirs(save_path_1)
    #                 save_path = save_path_1
    #             else:
    #                 save_path_2 = save_path + "_2"
    #                 if not os.path.exists(save_path_2):
    #                     os.makedirs(save_path_2)
    #                     save_path = save_path_2
    #
    #
    #         _ = [sio.imsave(save_path+"/"+name+"_recov.jpg", x) for x,name in zip(Recovered,file_names)]
    #         _ = [sio.imsave(save_path+"/"+name+"_noisy.jpg", x) for x,name in zip(Noisy,file_names)]
    #         Residual_Curve = np.array(Residual_Curve).mean(axis=0)
    #         np.save(save_path+"/"+"residual_curve.npy", Residual_Curve)
    #         np.save(save_path+"/original.npy", Original)
    #         np.save(save_path+"/recovered.npy", Recovered)
    #         np.save(save_path+"/noisy.npy", Noisy)
