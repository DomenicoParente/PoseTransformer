from imgaug import augmenters as iaa
import numpy as np


class ImageAugmentation:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.SomeOf((0, 5), [
                iaa.Sometimes(0.75,
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)),  # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)),
                        # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)),
                        # blur image using local medians with kernel sizes between 2 and 7
                        ])
                ),
                iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),  # sharpen images
                iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),
                iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                # add gaussian noise to images
                iaa.OneOf([
                    iaa.Dropout((0.01, 0.1), per_channel=0.5),  # randomly remove up to 10% of the pixels
                    iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                ]),
                iaa.Invert(0.05),  # invert color channels
                iaa.Add((-10, 10), per_channel=0.5),
                # change brightness of images (by -10 to 10 of original value)
                iaa.AddToHueAndSaturation((-20, 20)),  # change hue and saturation
                # either change the brightness of the whole image (sometimes
                # per channel) or change the brightness of subareas
                iaa.OneOf([
                    iaa.Multiply((0.5, 1.5), per_channel=0.5),
                    iaa.BlendAlphaSomeColors(iaa.TotalDropout(1.0))
                ]),
                iaa.LinearContrast((0.5, 2.0), per_channel=0.5),
                iaa.OneOf([
                    iaa.HistogramEqualization(),
                    iaa.GammaContrast(per_channel=True),
                    iaa.SigmoidContrast(cutoff=0.5)
                ]),
                iaa.OneOf([
                    iaa.SaltAndPepper((0.1, 0.3)),
                    iaa.CoarseSaltAndPepper((0.01, 0.1), size_percent=(0.01, 0.15)),
                    iaa.AddToHueAndSaturation((-50, 50), per_channel=True)
                ])
            ])
        ], random_order=True)

    def __call__(self, img):
        img = np.array(img)
        return self.aug.augment_image(img).copy()
