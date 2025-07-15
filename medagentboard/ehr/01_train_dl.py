import os
import argparse

import pandas as pd
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
import torch

from pyehr.datasets.utils.datamodule import EhrDataModule
from pyehr.pipelines import DlPipeline, MlPipeline
from pyehr.utils.bootstrap import run_bootstrap


def run_dl_experiment(config):
    # data
    dataset_path = f'my_datasets/ehr/{config["dataset"]}/processed/{config["split"]}'
    dm = EhrDataModule(dataset_path, task=config["task"], batch_size=config["batch_size"], test_mode="fusion")

    # los infomation
    los_info = pd.read_pickle(os.path.join(dataset_path, "los_info.pkl")) if config["task"] == "los" else None
    config["los_info"] = los_info

    # logger
    logger = CSVLogger(save_dir="logs", name=f'{config["dataset"]}/{config["task"]}', version=f"{config['model']}")

    # main metric
    main_metric = "auroc" if config["task"] in ["mortality", "readmission", "sptb"] else "mae"
    mode = "max" if config["task"] in ["mortality", "readmission", "sptb"] else "min"
    config["main_metric"] = main_metric

    # EarlyStop and checkpoint callback
    early_stopping_callback = EarlyStopping(monitor=main_metric, patience=config["patience"], mode=mode)
    checkpoint_callback = ModelCheckpoint(filename="best", monitor=main_metric, mode=mode)

    # seed for reproducibility
    L.seed_everything(42)

    # train/val/test
    pipeline = DlPipeline(config)
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = [0]
    else:
        accelerator = "cpu"
        devices = 1
    trainer = L.Trainer(accelerator=accelerator, devices=devices, max_epochs=config["epochs"], logger=logger, callbacks=[early_stopping_callback, checkpoint_callback])
    trainer.fit(pipeline, dm)

    # Load best model checkpoint
    best_model_path = checkpoint_callback.best_model_path
    print("best_model_path:", best_model_path)
    pipeline = DlPipeline.load_from_checkpoint(best_model_path, config=config)
    trainer.test(pipeline, dm)

    perf = pipeline.test_performance
    outs = pipeline.test_outputs
    return config, perf, outs


def run_ml_experiment(config):
    # data
    dataset_path = f'my_datasets/ehr/{config["dataset"]}/processed/{config["split"]}'
    dm = EhrDataModule(dataset_path, task=config["task"], batch_size=config["batch_size"])

    # los infomation
    los_info = pd.read_pickle(os.path.join(dataset_path, 'los_info.pkl'))
    config["los_info"] = los_info

    # logger
    logger = CSVLogger(save_dir="logs", name=f'{config["dataset"]}/{config["task"]}', version=f"{config['model']}")

    # main metric
    main_metric = "auroc" if config["task"] in ["mortality", "readmission"] else "mae"
    config["main_metric"] = main_metric

    # seed for reproducibility
    L.seed_everything(42)

    # train/val/test
    pipeline = MlPipeline(config)
    trainer = L.Trainer(accelerator="cpu", max_epochs=1, logger=logger, num_sanity_val_steps=0)
    trainer.fit(pipeline, dm)

    trainer.test(pipeline, dm)

    perf = pipeline.test_performance
    outs = pipeline.test_outputs
    return config, perf, outs


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate deep learning models for EHR data")

    # Basic configurations
    parser.add_argument("--model", "-m", type=str, nargs="+", required=True, help="Model name")
    parser.add_argument("--dataset", "-d", type=str, required=True, help="Dataset name", choices=["tjh", "mimic-iv", "esrd", "obstetrics"])
    parser.add_argument("--task", "-t", type=str, required=True, help="Task name", choices=["mortality", "readmission", "los", "sptb"])

    # Model and training hyperparameters
    parser.add_argument("--hidden_dim", "-hd", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", "-bs", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="Number of epochs")
    parser.add_argument("--patience", "-p", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--output_dim", "-od", type=int, default=1, help="Output dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators for tree-based models")
    parser.add_argument("--max_depth", type=int, default=10, help="Max depth for tree-based models")

    # Additional configurations
    parser.add_argument("--output_root", type=str, default="logs", help="Root directory for saving outputs")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()

    # Set up the configuration dictionary
    config = {
        "dataset": args.dataset,
        "task": args.task,
        "hidden_dim": args.hidden_dim,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "patience": args.patience,
        "output_dim": args.output_dim,
        "seed": args.seed,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
    }

    # Set the input dimensions based on the dataset
    if args.dataset == "mimic-iv":
        config["demo_dim"] = 2
        config["lab_dim"] = 42
        config["split"] = "split"
    elif args.dataset == "esrd":
        config["demo_dim"] = 0
        config["lab_dim"] = 17
        config["split"] = "split"
    elif args.dataset == "obstetrics":
        config["demo_dim"] = 0
        config["lab_dim"] = 32
        config["split"] = "solo"
    else:
        raise ValueError("Unsupported dataset. Choose either 'tjh' or 'mimic-iv' or 'esrd' or 'obstetrics'.")

    perf_all_df = pd.DataFrame()
    for model in args.model:
        # Add the model name to the configuration
        config["model"] = model

        # Print the configuration
        print("Configuration:")
        for key, value in config.items():
            print(f"{key}: {value}")

        # Run the experiment
        try:
            run_experiment = run_ml_experiment if model in ["CatBoost", "DT", "RF", "XGBoost"] else run_dl_experiment
            config, perf, outs = run_experiment(config)
        except Exception as e:
            print(f"Error occurred while running the experiment for model {model}.")
            print(e)
            continue

        print("Test samples:", len(outs['preds']))

        # Save the performance and outputs
        save_dir = os.path.join(args.output_root, f"{args.dataset}/{args.task}/{model}")
        os.makedirs(save_dir, exist_ok=True)

        # Run bootstrap
        perf_boot = run_bootstrap(outs['preds'], outs['labels'], config)
        for key, value in perf_boot.items():
            if args.task in ["mortality", "readmission", "sptb"]:
                perf_boot[key] = f'{value["mean"] * 100:.2f}±{value["std"] * 100:.2f}'
            else:
                perf_boot[key] = f'{value["mean"]:.2f}±{value["std"]:.2f}'

        # Save performance and outputs
        perf_boot = dict({
            "model": model,
            "dataset": args.dataset,
            "task": args.task,
        }, **perf_boot)
        perf_df = pd.DataFrame(perf_boot, index=[0])
        perf_df.to_csv(os.path.join(save_dir, "performance.csv"), index=False)
        pd.to_pickle(outs, os.path.join(save_dir, "outputs.pkl"))
        print(f"Performance and outputs saved to {save_dir}")

        # Append performance to the all performance DataFrame
        perf_all_df = pd.concat([perf_all_df, perf_df], ignore_index=True)

    # Save all performance
    try:
        perf_all_df.to_csv(os.path.join(args.output_root, f"{args.dataset}/{args.task}/all_performance.csv"), index=False)
        print(f"All performances saved to {os.path.join(args.output_root, f'{args.dataset}/{args.task}/all_performance.csv')}")
    except Exception as e:
        print(e)
    print("All experiments completed.")