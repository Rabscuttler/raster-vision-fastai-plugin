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

class PredictPV(rv.ExperimentSet):
    def exp_main(self, exp_id, config, raw_uri, root_uri, test=False):


        test = str_to_bool(test)
        exp_id = 'duke-seg2'
        debug = False
        chip_size = 300

        task = rv.TaskConfig.builder(rv.SEMANTIC_SEGMENTATION) \
            .predict_package_uri('/opd/data/rv/try2/duke-seg2/predict_package.zip')\
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
            .with_pretrained_model(uri)\
            .build()

        experiment = rv.ExperimentConfig.builder() \
                                        .with_id(exp_id) \
                                        .with_task(task) \
                                        .with_backend(backend) \
                                        .with_root_uri(root_uri) \
                                        .build()

        return experiment


if __name__ == '__main__':
    rv.main()
