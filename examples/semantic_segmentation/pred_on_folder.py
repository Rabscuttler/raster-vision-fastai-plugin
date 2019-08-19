import os
import glob


root_dir = '/opt/data/true/data'


# EPOCHS  = 20
# No Fine Tuning

pred_package = '/opt/data/rv/try2/bundle/duke-seg/predict_package.zip'
imgs = [15196, 13596, 13598, 4796, 3199]

for img in imgs:
    to_predict = '{}/{}.png'.format(root_dir, img)
    output = '/opt/data/true/output20/{}.tif'.format(img)
    os.system("rastervision -p fastai predict {} {} {}".format(pred_package, to_predict, output))
    print('Created prediction segmentation {}.tif'.format(img))



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



