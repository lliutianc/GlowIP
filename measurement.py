import numpy as np


class NoisyMeasurement:
    def __init__(self, distribution, channel=None, area=None):
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

    def __call__(self, images):
        return self._add_noise(images)

    def _add_noise(self, images):
        img_num_channel = images.shape[1]
        noise = self.distribution(images.shape)

        # add channel mask

        if self.channel:
            if self.channel != img_num_channel:
                channel_mask = np.zeros_like(noise)
                channel_mask[:, self.channel, ...] = 1.
                noise *= channel_mask

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

        return noise + images


def gaussian_noise(loc, scale):
    return lambda size: np.random.normal(size=size, loc=loc, scale=scale)