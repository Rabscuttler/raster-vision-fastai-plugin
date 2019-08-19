import os
from os.path import join
import random

import rastervision as rv
from examples.utils import get_scene_info, str_to_bool, save_image_crop

from fastai_plugin.chip_classification_backend_config import (
    FASTAI_CHIP_CLASSIFICATION)

# rastervision -p fastai run local -e examples.chip_classification.pv_classification -m *exp_resnet18_oversample* -a raw_uri /opt/data/labels2 -a processed_uri /opt/data/processed -a root_uri /opt/data/rv/test1 -a test True


class ChipClassificationExperiments(rv.ExperimentSet):
    def get_exp(self, exp_id, config, raw_uri, processed_uri, root_uri, test=False):
        """Chip classification experiment on Spacenet Rio dataset.
        Run the data prep notebook before running this experiment. Note all URIs can be
        local or remote.
        Args:
            raw_uri: (str) directory of raw data
            processed_uri: (str) directory of processed data
            root_uri: (str) root directory for experiment output
            test: (bool) if True, run a very small experiment as a test and generate
                debug output
        """
        # base_uri = '/opt/data/labels2'
        base_uri = raw_uri

        raster_uri = raw_uri  # rasters and labels in same directory for now
        label_uri = raw_uri

        # Find all of the image ids that have associated images and labels. Collect
        # these values to use as our scene ids.
        # TODO use PV Array dataframe to select these
        # label_paths = list_paths(label_uri, ext='.geojson')
        # scene_ids = [x.split('.')[-2].split('/')[-1] for x in label_paths]

        scene_ids = [
            'so9051_rgb_250_04', 'so9265_rgb_250_05', 'sp3590_rgb_250_04',
            'sj7304_rgb_250_04', 'su1385_rgb_250_06', 'st0709_rgb_250_05',
            'sj9004_rgb_250_05', 'st8022_rgb_250_05', 'st8303_rgb_250_05',
            'sj9402_rgb_250_05', 'so9078_rgb_250_06', 'sj9003_rgb_250_05',
            'sk0003_rgb_250_05', 'st8468_rgb_250_04', 'st6980_rgb_250_04',
            'su0883_rgb_250_05', 'su0983_rgb_250_05', 'so9249_rgb_250_05',
            'su1478_rgb_250_04', 'su1377_rgb_250_04', 'sj9002_rgb_250_06',
            'sj8903_rgb_250_04', 'sj9902_rgb_250_05', 'sj9602_rgb_250_05',
            'tg2827_rgb_250_04', 'sj9702_rgb_250_05', 'sj9803_rgb_250_04',
            'sj9802_rgb_250_05', 'sk0504_rgb_250_04', 'sk0302_rgb_250_05',
            'sk0306_rgb_250_04', 'sk0206_rgb_250_04', 'sk0207_rgb_250_04',
            'sk0503_rgb_250_04', 'sj9903_rgb_250_04', 'sk0202_rgb_250_06',
            'sk0309_rgb_250_03', 'sk0605_rgb_250_04', 'sk0405_rgb_250_04',
            'sk0404_rgb_250_04', 'sk0502_rgb_250_05', 'st5071_rgb_250_05',
            'sp3293_rgb_250_03', 'sy7691_rgb_250_05', 'sp3294_rgb_250_03',
            'sp3892_rgb_250_05', 'sp3690_rgb_250_04', 'st9979_rgb_250_05',
            'se6154_rgb_250_03', 'so8476_rgb_250_06', 'so8072_rgb_250_04',
            'so7972_rgb_250_04', 'sp3491_rgb_250_03', 'sp3490_rgb_250_03',
            'sp3291_rgb_250_03', 'sp3292_rgb_250_03', 'sp3492_rgb_250_03',
            'sk0212_rgb_250_03', 'so7878_rgb_250_06', 'tl1239_rgb_250_03',
            'su0972_rgb_250_03', 'st1532_rgb_250_04', 'so7556_rgb_250_05',
            'st7091_rgb_250_07', 'sn2040_rgb_250_04', 'so7371_rgb_250_04',
            'tl6064_rgb_250_05', 'so9255_rgb_250_05', 'st1826_rgb_250_04',
            'st1528_rgb_250_04', 'st1629_rgb_250_04', 'st0727_rgb_250_04',
            'st0827_rgb_250_04', 'st0928_rgb_250_04', 'st0930_rgb_250_04',
            'st0929_rgb_250_04', 'st0832_rgb_250_05', 'tl1750_rgb_250_03',
            'st2322_rgb_250_05', 'st1623_rgb_250_04', 'st1523_rgb_250_04',
            'st1624_rgb_250_04', 'st1424_rgb_250_04', 'st1421_rgb_250_05',
            'sp3793_rgb_250_04', 'sp3792_rgb_250_04', 'sj9912_rgb_250_03',
            'sk2347_rgb_250_05', 'sp3391_rgb_250_03', 'tl1846_rgb_250_03',
            'sp5177_rgb_250_03', 'sn3251_rgb_250_04', 'sp3693_rgb_250_04',
            'st2014_rgb_250_06', 'st2015_rgb_250_06', 'st2115_rgb_250_05',
            'st2114_rgb_250_05', 'sn4257_rgb_250_04', 'su4223_rgb_250_04',
            'su4323_rgb_250_04', 'tl3068_rgb_250_04', 'sp5178_rgb_250_03',
            'sp3791_rgb_250_04', 'st3689_rgb_250_03', 'st3789_rgb_250_03',
            'st0411_rgb_250_04', 'st0212_rgb_250_04', 'st0112_rgb_250_04',
            'st0211_rgb_250_04', 'st0111_rgb_250_04', 'st0209_rgb_250_05',
            'st0210_rgb_250_05', 'sj6714_rgb_250_04', 'sp3893_rgb_250_05',
            'su6712_rgb_250_04', 'su6713_rgb_250_04', 'st9363_rgb_250_04',
            'st9463_rgb_250_04', 'nr3059_rgb_250_03', 'st8576_rgb_250_03',
            'sp7948_rgb_250_04', 'sp6138_rgb_250_07', 'tl2276_rgb_250_04',
            'sm9817_rgb_250_04', 'sm9816_rgb_250_04', 'sm9716_rgb_250_04',
            'sm9616_rgb_250_04', 'sm9818_rgb_250_04', 'sm9009_rgb_250_04',
            'sm9721_rgb_250_05', 'sm9720_rgb_250_05', 'sm9101_rgb_250_04',
            'sm9201_rgb_250_04', 'sm9010_rgb_250_04', 'sm9109_rgb_250_04',
            'sn6502_rgb_250_04', 'sn6601_rgb_250_04', 'sn6201_rgb_250_04',
            'sn6202_rgb_250_04', 'st6788_rgb_250_05', 'st6688_rgb_250_05',
            'st6689_rgb_250_06', 'su0807_rgb_250_05', 'su0806_rgb_250_05',
            'sz0998_rgb_250_05', 'sz1099_rgb_250_05', 'su3743_rgb_250_04',
            'su3744_rgb_250_04', 'su6509_rgb_250_04', 'su6409_rgb_250_04',
            'su6410_rgb_250_04', 'su5413_rgb_250_04', 'su2088_rgb_250_04',
            'su5703_rgb_250_04', 'su5603_rgb_250_04', 'su5604_rgb_250_04',
            'st7642_rgb_250_06', 'st7744_rgb_250_05', 'st6728_rgb_250_05',
            'st8558_rgb_250_04', 'st2735_rgb_250_04', 'tl4990_rgb_250_05',
            'sm7209_rgb_250_04', 'st8864_rgb_250_04', 'tg5013_rgb_250_04',
            'st1198_rgb_250_04', 'st1298_rgb_250_04', 'st1722_rgb_250_04',
            'tq1078_rgb_250_05', 'su6401_rgb_250_04', 'st8753_rgb_250_04',
            'st8455_rgb_250_05', 'st8660_rgb_250_04', 'st8760_rgb_250_04',
            'st8765_rgb_250_04', 'sp7638_rgb_250_05', 'tl6332_rgb_250_04',
            'st8705_rgb_250_05', 'sy3297_rgb_250_06', 'sy3498_rgb_250_06',
            'se3636_rgb_250_01', 'st6578_rgb_250_05', 'st6478_rgb_250_05',
            'st5479_rgb_250_06', 'se2931_rgb_250_02', 'sd6835_rgb_250_01',
            'st2228_rgb_250_05', 'st2227_rgb_250_05']

        # Experiment label and root directory for output
        exp_id = 'pv-classification'
        root_uri = root_uri

        # num_steps = 1e4 # 1e5 takes too long
        num_epochs = 20
        batch_size = 16
        debug = True

        # Split the data into training and validation sets:
        # Randomize the order of all scene ids
        random.seed(5678)
        scene_ids = sorted(scene_ids)
        random.shuffle(scene_ids)

        # Set scenes
        num_train_ids = round(len(scene_ids) * 0.8)
        train_ids = scene_ids[0:num_train_ids]
        val_ids = scene_ids[num_train_ids:]
        # train_ids = scene_ids
        # val_ids = scene_ids

        chip_key = 'pv-chip-classification'
        if test:
            exp_id += '-test'
            config['num_epochs'] = 1
            config['batch_sz'] = 8
            config['debug'] = True
            train_scene_info = scene_ids[0:1]
            val_scene_info = scene_ids[0:1]

        task = rv.TaskConfig.builder(rv.CHIP_CLASSIFICATION) \
                            .with_chip_size(200) \
                            .with_classes({
                                'pv': (1, 'yellow'),
                                'background': (2, 'black')
                            }) \
                            .build()

        backend = rv.BackendConfig.builder(FASTAI_CHIP_CLASSIFICATION) \
                                  .with_task(task) \
                                  .with_train_options(**config) \
                                  .build()

        def make_scene(id):
            image_uri = os.path.join(base_uri, '{}.jpg'.format(id))
            label_uri = os.path.join(base_uri, '{}.geojson'.format(id))

            # aoi_uri = join(raw_uri, aoi_path)

            if test:
                crop_uri = join(processed_uri, 'crops', id + '.jpg')
                save_image_crop(image_uri, crop_uri, label_uri=label_uri, size=600, min_features=1)
                img_uri = crop_uri

            label_source = rv.LabelSourceConfig.builder(rv.CHIP_CLASSIFICATION) \
                                               .with_uri(label_uri) \
                                               .with_ioa_thresh(0.5) \
                                               .with_use_intersection_over_cell(False) \
                                               .with_pick_min_class_id(True) \
                                               .with_background_class_id(2) \
                                               .with_infer_cells(True) \
                                               .build()

            # .with_aoi_uri(aoi_uri) \
            return rv.SceneConfig.builder() \
                                 .with_task(task) \
                                 .with_id(id) \
                                 .with_raster_source(image_uri) \
                                 .with_label_source(label_source) \
                                 .build()

        train_scenes = [make_scene(id) for id in train_ids]
        val_scenes = [make_scene(id) for id in val_ids]

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id(exp_id) \
                                        .with_chip_key(chip_key) \
                                        .with_root_uri(root_uri) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .build()

        return experiment

    def exp_resnet18(self, raw_uri, processed_uri, root_uri, test=False):
        exp_id = 'resnet18'
        config = {
            'batch_sz': 8,
            'num_epochs': 5,
            'debug': False,
            'lr': 1e-4,
            'sync_interval': 10,
            'model_arch': 'resnet18',
            'flip_vert': True,
            'oversample': {
                'rare_class_ids': [1],
                'rare_target_prop': 1.0
            },
        }
        return self.get_exp(exp_id, config, raw_uri, processed_uri, root_uri, test)

    def exp_resnet18_oversample(self, raw_uri, processed_uri, root_uri, test=False):
        exp_id = 'resnet18'
        config = {
            'batch_sz': 8,
            'num_epochs': 5,
            'debug': False,
            'lr': 1e-4,
            'sync_interval': 10,
            'model_arch': 'resnet18',
            'flip_vert': True,
            'oversample': {
                'rare_class_ids': [1],
                'rare_target_prop': 1.0
            },
        }
        return self.get_exp(exp_id, config, raw_uri, processed_uri, root_uri, test)


    def exp_resnet50(self, raw_uri, processed_uri, root_uri, test=False):
        exp_id = 'resnet50'
        config = {
            'batch_sz': 8,
            'num_epochs': 5,
            'debug': False,
            'lr': 1e-4,
            'sync_interval': 10,
            'model_arch': 'resnet50',
        }
        return self.get_exp(exp_id, config, raw_uri, processed_uri, root_uri, test)


if __name__ == '__main__':
    rv.main()
