import sys
sys.path.append('..')

import h5py

from caption import utils

dt = h5py.special_dtype(vlen=str)
# print('Creating training dataset...')
#
# ftrain = h5py.File('coco/train2014.hdf5', 'w')
# train_paths = utils.get_image_paths('coco/train2014')
# captions = utils.get_captions('coco/train2014/captions.txt', train_paths)
#
# size = (128, 128)
#
# im_set = ftrain.create_dataset('images', (len(train_paths), size[0], size[1], 3), dtype='f')
# cap_set = ftrain.create_dataset('captions', (len(captions),), dtype=dt)
#
# for i, tp, c in zip(range(len(train_paths)), train_paths, captions):
#     print('%s\t%s' % (tp, c[:50]))
#     mat = utils.load_image(tp, size=size)
#     im_set[i, :, :, :] = mat
#     cap_set[i] = c
#
# ftrain.close()


print('Creating validation dataset...')

fval = h5py.File('coco/val2014.hdf5', 'w')
val_paths = utils.get_image_paths('coco/val2014')
captions = utils.get_captions('coco/val2014/captions.txt', val_paths)

size = (128, 128)

im_set = fval.create_dataset('images', (len(val_paths), size[0], size[1], 3), dtype='f')
cap_set = fval.create_dataset('captions', (len(captions),), dtype=dt)

for i, tp, c in zip(range(len(val_paths)), val_paths, captions):
    print('%s\t%s' % (tp, c[:50]))
    mat = utils.load_image(tp, size=size)
    im_set[i, :, :, :] = mat
    cap_set[i] = c

fval.close()
