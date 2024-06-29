from pathlib import Path
from typing import Dict

import ray
import torch
from filelock import FileLock
from loguru import logger
from mltrainer import ReportTypes, Trainer, TrainerSettings
from mads_datasets.base import BaseDatastreamer
from mltrainer.preprocessors import BasePreprocessor
import numpy as np
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from src import datasets, models, metrics

SAMPLE_INT = tune.search.sample.Integer
SAMPLE_FLOAT = tune.search.sample.Float


def train(config: Dict):
    """
    The train function should receive a config file, which is a Dict
    ray will modify the values inside the config before it is passed to the train
    function.
    """

    # data_dir wordt doorgegeven via tune.with_parameters
    data_dir = config["data_dir"]
    trainfile = data_dir / 'heart_big_train.parq'
    testfile = data_dir / 'heart_big_test.parq'
    
    shape = (16, 12)
    traindataset = datasets.HeartDataset2D(trainfile, target="target", shape=shape)
    testdataset = datasets.HeartDataset2D(testfile, target="target", shape=shape)

    with FileLock(data_dir / ".lock"):
        # we lock the datadir to avoid parallel instances trying to
        # access the datadir
        train = datasets.BaseDatastreamer(traindataset, preprocessor=BasePreprocessor(), batchsize=32)
        valid = datasets.BaseDatastreamer(testdataset, preprocessor=BasePreprocessor(), batchsize=32)
    
    device = "cpu"
    traindataset.to(device)
    testdataset.to(device)


    # we set up the metric
    f1micro = metrics.F1Score(average='micro')
    f1macro = metrics.F1Score(average='macro')
    precision = metrics.Precision('micro')
    recall = metrics.Recall('macro')
    accuracy = metrics.Accuracy()
    model = models.CNN(config)
    model.to(device)

    trainersettings = TrainerSettings(
        epochs=10,
        metrics=[accuracy, f1micro, f1macro, precision, recall],
        logdir=Path("."),
        train_steps=len(train),  # type: ignore
        valid_steps=len(valid),  # type: ignore
        reporttypes=[ReportTypes.RAY],
        scheduler_kwargs={"factor": 0.5, "patience": 5},
        earlystop_kwargs=None,
    )

    trainer = Trainer(
        model=model,
        settings=trainersettings,
        loss_fn=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam,
        traindataloader=trainstreamer.stream(),
        validdataloader=teststreamer.stream(),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

    trainer.loop()

if __name__ == "__main__":
    ray.init()

    data_dir = Path("data").resolve()
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        logger.info(f"Created {data_dir}")
    tune_dir = Path("models/ray").resolve()

    config = {
        "hidden": tune.randint(16, 64),
        "num_layers": tune.randint(2, 5),
        "num_classes": 5,
        "dropout_rate": tune.uniform(0.0, 0.3),
        "shape": (16, 12),
    }

    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")
    reporter.add_metric_column("f1micro")
    reporter.add_metric_column("f1macro")
    reporter.add_metric_column("precision")
    reporter.add_metric_column("Recall")
    
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=50,
        reduction_factor=3,
        stop_last_trials=False,
    )

    bohb_search = TuneBOHB()

    analysis = tune.run(
        train,
        config=config,
        metric="test_loss",
        mode="min",
        progress_reporter=reporter,
        storage_path=str(tune_dir),
        num_samples=50,
        search_alg=bohb_search,
        scheduler=bohb_hyperband,
        verbose=1,
    )

    ray.shutdown()
