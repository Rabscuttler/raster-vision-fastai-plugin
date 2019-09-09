import os
import glob

# EPOCHS  = 500
# No Fine Tuning

# root_dir = '/opt/data/true/data'
pred_package = '/opt/data/rv/try2/bundle/duke-seg3/predict_package.zip'
# images = [x.split('/')[-1].strip('.png') for x in glob.glob(root_dir + '/*.png')]
# #
# for img in images:
#     to_predict = '{}/{}.png'.format(root_dir, img)
#     output = '/opt/data/true/data_preds/{}.tif'.format(img)
#     os.system("rastervision -p fastai predict {} {} {}".format(pred_package, to_predict, output))
#     print('Created prediction segmentation {}.tif'.format(img))

# towns = ['bristol' 'cambridge', 'knowsley', 'liverpool', 'manchester', 'peterborough']
# towns = ['bristol' 'cambridge']
towns = ['bristol']

for town in towns:
    root_dir = '/opt/data/true/{}'.format(town)
    images = [x.split('/')[-1].strip('.jpg') for x in glob.glob(root_dir + '/*.jpg')]
    #
    for img in images:
        to_predict = '{}/{}.jpg'.format(root_dir, img)
        output = '/opt/data/true/{}_preds/{}.tif'.format(town, img)
        os.system("rastervision -p fastai predict {} {} {}".format(pred_package, to_predict, output))
        print('Created prediction segmentation {} {}.tif'.format(town, img))


