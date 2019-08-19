import re
import random
import os
from os.path import join
import glob

import rastervision as rv
from examples.utils import str_to_bool

from fastai_plugin.semantic_segmentation_backend_config import (
    FASTAI_SEMANTIC_SEGMENTATION)

# Run experiment
# rastervision -p fastai run local -e examples.semantic_segmentation.duke -m *exp_resnet18* -a raw_uri /opt/data/ -a root_uri /opt/data/rv/try2 -a test True

# Check tensorboard
# tensorboard --logdir /opt/data/rv/

# Raster Vision can run certain commands in parallel, such as theCHIPandPREDICTcommands.
# To do so, use the–splitsoption in theruncommand of the CLI.
# Use splits to speed things up!

class DukePV(rv.ExperimentSet):
    def exp_main(self, exp_id, config, raw_uri, root_uri, test=False):
        """Run an experiment on the Spacenet Vegas building dataset.

        This is a simple example of how to do semantic segmentation on data that
        doesn't require any pre-processing or special permission to access.

        Args:
            raw_uri: (str) directory of raw data (the root of the Spacenet dataset)
            root_uri: (str) root directory for experiment output
            test: (bool) if True, run a very small experiment as a test and generate
                debug output

        """
        base_uri = join(
            raw_uri, 'duke')
        raster_uri = base_uri
        label_uri = join(raw_uri, 'duke_labels')
        scene_ids = [os.path.basename(x).replace('.geojson', '') for x in glob.glob(join(label_uri, '*.geojson'))]

        random.seed(5678)
        scene_ids = sorted(scene_ids)
        random.shuffle(scene_ids)

        num_train_ids = int(len(scene_ids) * 0.8)
        train_ids = scene_ids[0:num_train_ids]
        val_ids = scene_ids[num_train_ids:]

        test = str_to_bool(test)
        exp_id = 'duke-seg3'
        debug = False
        chip_size = 300
        if test:
            exp_id += '-test'
            train_ids = ['11ska355800']
            val_ids = ['11ska490710']
            config['debug'] = False
            config['batch_sz'] = 1
            config['num_epochs'] = 1

        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                            .with_chip_size(chip_size) \
                            .with_classes({
                                'PV': (1, 'yellow'),
                                'Background': (2, 'black')
                            }) \
                            .with_chip_options(
                                chips_per_scene=9,
                                debug_chip_probability=0.25,
                                negative_survival_probability=0.1,
                                target_classes=[1],
                                target_count_threshold=1000) \
                            .build()

        backend = rv.BackendConfig.builder(FASTAI_SEMANTIC_SEGMENTATION) \
                                  .with_task(task) \
                                  .with_train_options(**config) \
                                  .build()

        def make_scene(id):
            train_image_uri = os.path.join(raster_uri,
                                           '{}.tif'.format(id))

            raster_source = rv.RasterSourceConfig.builder(rv.RASTERIO_SOURCE) \
                .with_uri(train_image_uri) \
                .with_channel_order([0, 1, 2]) \
                .with_stats_transformer() \
                .build()

            vector_source = os.path.join(
                label_uri, '{}.geojson'.format(id))
            label_raster_source = rv.RasterSourceConfig.builder(rv.RASTERIZED_SOURCE) \
                .with_vector_source(vector_source) \
                .with_rasterizer_options(2) \
                .build()

            label_source = rv.LabelSourceConfig.builder(rv.SEMANTIC_SEGMENTATION) \
                .with_raster_source(label_raster_source) \
                .build()

            scene = rv.SceneConfig.builder() \
                .with_task(task) \
                .with_id(id) \
                .with_raster_source(raster_source) \
                .with_label_source(label_source) \
                .build()

            return scene

        train_scenes = [make_scene(id) for id in train_ids]
        val_scenes = [make_scene(id) for id in val_ids]

        dataset = rv.DatasetConfig.builder() \
            .with_train_scenes(train_scenes) \
            .with_validation_scenes(val_scenes) \
            .build()

        analyzer = rv.AnalyzerConfig.builder(rv.STATS_ANALYZER) \
                                    .with_sample_prob(0.1) \
                                    .build()

        # Need to use stats_analyzer because imagery is uint16.
        experiment = rv.ExperimentConfig.builder() \
                                        .with_id(exp_id) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_analyzer(analyzer) \
                                        .with_dataset(dataset) \
                                        .with_root_uri(root_uri) \
                                        .build()

        return experiment

    def exp_resnet18(self, raw_uri, root_uri, test=False):
        # A set of hyperparams that result in greater accuracy.
        exp_id = 'resnet18'
        config = {
            'batch_sz': 8,
            'num_epochs': 500,
            'debug': False,
            'lr': 1e-5,
            'one_cycle': True,
            'sync_interval': 1,
            'tta': True,
            'model_arch': 'resnet18',
            'flip_vert': True
        }
        return self.exp_main(exp_id, config, raw_uri, root_uri, test)

    def exp_resnet18_oversample(self, raw_uri, root_uri, test=False):
        # A set of hyperparams that result in greater accuracy.
        exp_id = 'resnet18_oversample'
        config = {
            'batch_sz': 8,
            'num_epochs': 20,
            'debug': False,
            'lr': 1e-4,
            'sync_interval': 1,
            'tta': True,
            'model_arch': 'resnet18',
            'flip_vert': True,
            'oversample': {
                'rare_class_ids': [1],
                'rare_target_prop': 1.0
            }
        }
        return self.exp_main(exp_id, config, raw_uri, root_uri, test)

if __name__ == '__main__':
    rv.main()
