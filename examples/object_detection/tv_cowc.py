import os
from os.path import join

import rastervision as rv
from examples.utils import str_to_bool, save_image_crop

from fastai_plugin.tv_object_detection_backend_config import (
    TV_OBJECT_DETECTION)


class CowcObjectDetectionExperiments(rv.ExperimentSet):
    def exp_main(self, raw_uri, processed_uri, root_uri, test=False):
        """Object detection on COWC (Cars Overhead with Context) Potsdam dataset

        Args:
            raw_uri: (str) directory of raw data
            processed_uri: (str) directory of processed data
            root_uri: (str) root directory for experiment output
            test: (bool) if True, run a very small experiment as a test and generate
                debug output
        """
        test = str_to_bool(test)
        exp_id = 'cowc-object-detection'
        num_epochs = 20
        batch_sz = 16
        debug = False
        lr = 1e-4
        model_arch = 'resnet18'
        sync_interval = 10
        one_cycle = True
        train_scene_ids = ['2_10', '2_11', '2_12', '2_14', '3_11',
                           '3_13', '4_10', '5_10', '6_7', '6_9']
        val_scene_ids = ['2_13', '6_8', '3_10']

        if test:
            exp_id += '-test'
            num_epochs = 3
            batch_sz = 1
            debug = True
            train_scene_ids = train_scene_ids[0:1]
            val_scene_ids = val_scene_ids[0:1]

        # XXX set neg_ratio to 0 for testing purposes
        # since fastai can't handle neg chips afaik.
        task = rv.TaskConfig.builder(rv.OBJECT_DETECTION) \
                            .with_chip_size(300) \
                            .with_classes({'vehicle': (1, 'red')}) \
                            .with_chip_options(neg_ratio=0.0,
                                               ioa_thresh=0.8) \
                            .with_predict_options(merge_thresh=0.3,
                                                  score_thresh=0.5) \
                            .build()

        config = {
            'batch_sz': batch_sz,
            'num_epochs': num_epochs,
            'debug': debug,
            'lr': lr,
            'one_cycle': one_cycle,
            'sync_interval': sync_interval,
            'model_arch': model_arch
        }
        backend = rv.BackendConfig.builder(TV_OBJECT_DETECTION) \
                                  .with_task(task) \
                                  .with_train_options(**config) \
                                  .build()

        def make_scene(id):
            raster_uri = join(
                raw_uri, '4_Ortho_RGBIR/top_potsdam_{}_RGBIR.tif'.format(id))
            label_uri = join(
                processed_uri, 'labels', 'all', 'top_potsdam_{}_RGBIR.json'.format(id))

            if test:
                crop_uri = join(
                    processed_uri, 'crops', os.path.basename(raster_uri))
                save_image_crop(raster_uri, crop_uri, label_uri=label_uri,
                                size=1000, min_features=5)
                raster_uri = crop_uri

            return rv.SceneConfig.builder() \
                                 .with_id(id) \
                                 .with_task(task) \
                                 .with_raster_source(raster_uri, channel_order=[0, 1, 2]) \
                                 .with_label_source(label_uri) \
                                 .build()

        train_scenes = [make_scene(id) for id in train_scene_ids]
        val_scenes = [make_scene(id) for id in val_scene_ids]

        dataset = rv.DatasetConfig.builder() \
                                  .with_train_scenes(train_scenes) \
                                  .with_validation_scenes(val_scenes) \
                                  .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id(exp_id) \
                                        .with_root_uri(root_uri) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_dataset(dataset) \
                                        .build()

        return experiment


if __name__ == '__main__':
    rv.main()
