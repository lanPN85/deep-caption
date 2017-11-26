# VGG-Net setups

VGG = [
    {
        'filters': 64, 'kernel': 3, 'strides': 1
    },
    {
        'filters': 64, 'kernel': 3, 'strides': 1, 'pool': 2
    },
    {
        'filters': 128, 'kernel': 3, 'strides': 1
    },
    {
        'filters': 128, 'kernel': 3, 'strides': 1, 'pool': 2
    },
    {
        'filters': 256, 'kernel': 3, 'strides': 1
    },
    {
        'filters': 256, 'kernel': 3, 'strides': 1
    },
    {
        'filters': 256, 'kernel': 3, 'strides': 1, 'pool': 2
    },
    {
        'filters': 512, 'kernel': 3, 'strides': 1
    },
    {
        'filters': 512, 'kernel': 3, 'strides': 1
    },
    {
        'filters': 512, 'kernel': 3, 'strides': 1, 'pool': 2
    },
    {
        'filters': 512, 'kernel': 3, 'strides': 1
    },
    {
        'filters': 512, 'kernel': 3, 'strides': 1
    },
    {
        'filters': 512, 'kernel': 3, 'strides': 1, 'pool': 2, 'dense': 4096
    },
]

VGG_TRUNC = VGG[:-4] + [{'filters': 512, 'kernel': 3, 'strides': 1, 'pool': 2, 'dense': 4096}]
VGG_TRUNC_NODENSE = VGG[:-3]
