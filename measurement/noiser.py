import numpy as np
import torch

from measurement.measurement import Measurement
from glow.glow import Glow
from dcgan.dcgan import Generator


class NoisyMeasurement(Measurement):
    def __init__(self, distribution, channel=None, area=None, device='cpu'):
        """
        distribution: (partial) function to generate noise of specific sizes,
        channel: specify the first C channels to add noise,
        area: specify the area (as a quadrant) to add noise.
        """
        if area:
            if area not in [1, 2, 3, 4]:
                raise NotImplementedError(f'Cannot add noise to {area} now.')

        self.distribution = distribution
        self.area = area
        self.channel = channel
        self.device = device

    def forward(self, images):
        img_num_channel = images.shape[1]
        noise = self.distribution(images.shape)

        # add channel mask

        if self.channel:
            if self.channel != img_num_channel:
                channel_mask = np.zeros_like(noise)
                channel_mask[:, self.channel, ...] = 1.
                noise *= channel_mask

        # add area mask

        if self.area:
            img_height, img_width = images.shape[-2:]
            mask_lower = int(self.area in [3, 4])
            mask_right = int(self.area in [1, 4])

            mask_row_s = 0 + (img_height // 2) * mask_lower
            mask_row_e = (img_height// 2) * (1 + mask_lower)
            mask_col_s = 0 + (img_width // 2) * mask_right
            mask_col_e = (img_width // 2) * (1 + mask_right)

            area_mask = np.zeros(shape=(img_height, img_width))
            area_mask[mask_row_s:mask_row_e, mask_col_s:mask_col_e] = 1.
            noise *= area_mask

        noise = torch.tensor(noise, dtype=torch.float, requires_grad=False, device=self.device)
        return noise


def gaussian_noise(loc, scale):
    return lambda size: np.random.normal(size=size, loc=loc, scale=scale)


def image_noise(unused_loc, scale, **image_prior):
    noise = image_prior.get('noise', 'glow')
    size = image_prior.get('size')
    configs = image_prior.get('configs')
    device = image_prior.get('device')
    dataset = image_prior.get('dataset')

    if noise == 'glow':
        modeldir = f"./trained_models/{dataset}/glow-cs-{size}"

        glow = Glow((3, size, size),
                    K=configs["K"], L=configs["L"],
                    coupling=configs["coupling"],
                    n_bits_x=configs["n_bits_x"],
                    nn_init_last_zeros=configs["last_zeros"],
                    device=device)

        glow.load_state_dict(torch.load(modeldir + "/glowmodel.pt", map_location=device))
        glow.eval()

        n = size * size * 3

        def _image_noise(sample_size):
            bsz = sample_size[0]
            z = torch.normal(torch.zeros(size=(bsz, n)), torch.ones(size=(bsz, n)),
                             requires_grad=False)
            z_unflat = glow.unflatten_z(z, clone=False)

            noise = glow(z_unflat, reveres=True, reverse_clone=False)
            noise = glow.postprocess(noise, floor_clamp=False) * scale

            return noise

        return _image_noise

    elif noise == 'dcgan':
        raise NotImplementedError()
    else:
        raise NotImplementedError()






    return


    # use glow/gan to generated a face as a noise.
    # `scale` determines the intensity of the added image.
    pass