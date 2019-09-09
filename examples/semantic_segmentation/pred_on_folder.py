import os
import glob


# root_dir = '/opt/data/true/data'
# imgs = [15196, 13596, 13598, 4796, 3199]

# EPOCHS  = 20
# No Fine Tuning
# pred_package = '/opt/data/rv/try2/bundle/duke-seg/predict_package.zip'
# for img in imgs:
#     to_predict = '{}/{}.png'.format(root_dir, img)
#     output = '/opt/data/true/output20/{}.tif'.format(img)
#     os.system("rastervision -p fastai predict {} {} {}".format(pred_package, to_predict, output))
#     print('Created prediction segmentation {}.tif'.format(img))


# EPOCHS  = 100
# No Fine Tuning
# pred_package = '/opt/data/rv/try2/bundle/duke-seg2/predict_package.zip'
# imgs = [x.split('/')[-1].strip('.png') for x in glob.glob(root_dir + '/*.png')]
#
# for img in imgs:
#     to_predict = '{}/{}.png'.format(root_dir, img)
#     output = '/opt/data/true/output100/{}.tif'.format(img)
#     os.system("rastervision -p fastai predict {} {} {}".format(pred_package, to_predict, output))
#     print('Created prediction segmentation {}.tif'.format(img))


# EPOCHS  = 500
# No Fine Tuning
pred_package = '/opt/data/rv/try2/bundle/duke-seg3/predict_package.zip'
# images = [x.split('/')[-1].strip('.png') for x in glob.glob(root_dir + '/*.png')]
#
# for img in imgs:
#     to_predict = '{}/{}.png'.format(root_dir, img)
#     output = '/opt/data/true/output500/{}.tif'.format(img)
#     os.system("rastervision -p fastai predict {} {} {}".format(pred_package, to_predict, output))
#     print('Created prediction segmentation {}.tif'.format(img))


root_dir = '/opt/data/true/cant'
images = [x.split('/')[-1].strip('.jpg') for x in glob.glob(root_dir + '/*.jpg')]

for img in images:
    to_predict = '{}/{}.jpg'.format(root_dir, img)
    output = '/opt/data/true/output_all/{}.tif'.format(img)
    os.system("rastervision -p fastai predict {} {} {}".format(pred_package, to_predict, output))
    print('Created prediction segmentation {}.tif'.format(img))

