from pathlib import Path
from typing import Dict

import ray
import torch
from filelock import FileLock
from loguru import logger
from mltrainer import ReportTypes, Trainer, TrainerSettings
from mads_datasets.base import BaseDatastreamer
from mltrainer.preprocessors import BasePreprocessor
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers.hb_bohb import HyperBandForBOHB
from ray.tune.search.bohb import TuneBOHB
from src import datasets, models, metrics

# Definitions for shorthand use of tune's sampling methods for hyperparameters
SAMPLE_INT = tune.search.sample.Integer
SAMPLE_FLOAT = tune.search.sample.Float

def train(config: Dict):
    """
    Train function defines the training process configured to run with Ray Tune,
    which manages hyperparameter tuning dynamically.

    Parameters:
    - config (Dict): Dictionary containing configuration parameters and hyperparameters.
    """
    # Data directory from the config used for storing dataset files
    data_dir = config["data_dir"]
    
    # Specifying file paths for the training and testing datasets
    trainfile = data_dir / 'heart_big_train.parq'
    testfile = data_dir / 'heart_big_test.parq'
    
    # Setting the shape of the data as expected by the network
    shape = (16, 12)
    # Loading the datasets
    traindataset = datasets.HeartDataset2D(trainfile, target="target", shape=shape)
    testdataset = datasets.HeartDataset2D(testfile, target="target", shape=shape)

    # Ensuring only one instance accesses the data directory at a time
    with FileLock(data_dir / ".lock"):
        # Data streamers for managing data loading and preprocessing
        train = BaseDatastreamer(traindataset, preprocessor=BasePreprocessor(), batchsize=32)
        valid = BaseDatastreamer(testdataset, preprocessor=BasePreprocessor(), batchsize=32)
    
    # Defaulting to CPU usage for model training
    device = "cpu"
    
    # Class weights to handle imbalanced data in training
    class_weights = torch.tensor([1.0, 3.0, 3.0, 3.0, 3.0]) 

    # Defining a weighted loss function to take class imbalances into account
    loss_fn = models.WeightedCrossEntropyLoss(class_weights)

    # Setup of evaluation metrics for the training
    f1micro = metrics.F1Score(average='micro')
    f1macro = metrics.F1Score(average='macro')
    precision = metrics.Precision('micro')
    recall = metrics.Recall('macro')
    accuracy = metrics.Accuracy()
    # Initializing the model from the models module
    model = models.CNN(config)
    # Moving the model to the appropriate device (CPU or GPU)
    model.to(device)

    # Configuration for the training process including epochs, metrics, etc.
    trainersettings = TrainerSettings(
        epochs=10,
        metrics=[accuracy, f1micro, f1macro, precision, recall],
        logdir=Path("."),
        train_steps=len(train),  
        valid_steps=len(valid),  
        reporttypes=[ReportTypes.RAY],
        scheduler_kwargs={"factor": 0.5, "patience": 4},
        earlystop_kwargs=None,
    )

    # Initializing the trainer with the model and training settings
    trainer = Trainer(
        model=model,
        settings=trainersettings,
        loss_fn=loss_fn,
        optimizer=torch.optim.Adam,
        traindataloader=train.stream(),
        validdataloader=valid.stream(),
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau,
    )

    # Starting the training loop
    trainer.loop()

if __name__ == "__main__":
    # Initializing Ray for distributed computation
    ray.init()

    # Setting up directories for data and model output
    data_dir = Path("data").resolve()
    if not data_dir.exists():
        data_dir.mkdir(parents=True)
        logger.info(f"Created {data_dir}")
    tune_dir = Path("models/ray").resolve()
    
    # Configuration for hyperparameter tuning
    config = {
        "hidden": tune.randint(16, 64),
        "num_layers": tune.randint(1, 3),
        "tune_dir": tune_dir,
        "data_dir": data_dir,
        "num_classes": 5,
        "dropout": tune.uniform(0.01, 0.3),
        "shape": (16, 12),
    }

    # Reporter for logging progress during training
    reporter = CLIReporter()
    reporter.add_metric_column("Accuracy")
    reporter.add_metric_column("f1micro")
    reporter.add_metric_column("f1macro")
    reporter.add_metric_column("precision")
    reporter.add_metric_column("Recall")
    
    # Setting up hyperparameter tuning with HyperBand and BOHB
    bohb_hyperband = HyperBandForBOHB(
        time_attr="training_iteration",
        max_t=50,
        reduction_factor=3,
        stop_last_trials=False,
    )

    bohb_search = TuneBOHB()

    # Executing the training runs with Ray Tune
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

    # Clean up Ray resources after completion
    ray.shutdown()
