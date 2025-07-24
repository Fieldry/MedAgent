import os
import argparse
import shutil

import pandas as pd
import torch
import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from pyehr.datasets.utils.datamodule import EhrDataModule
from pyehr.pipelines import DlPipeline


def run_single_boost_round(config, boost_round, sample_weights=None):
    """
    Runs a single training round for one member of the boosting ensemble.
    """
    print(f"\n{'='*20} Starting Boosting Round {boost_round + 1}/{config['n_estimators']} {'='*20}")

    # --- Data ---
    dataset_path = f'my_datasets/ehr/{config["dataset"]}/processed/fold_{config["fold"]}'
    dm = EhrDataModule(dataset_path, task=config["task"], batch_size=config["batch_size"], train_sample_weights=sample_weights)

    # --- Logger ---
    version_str = f"{config['model']}_boost_round_{boost_round}"
    logger = CSVLogger(save_dir="logs", name=f'{config["dataset"]}/{config["task"]}', version=version_str)

    # --- Callbacks ---
    main_metric = config["main_metric"]
    mode = "max" if main_metric == "auroc" else "min"
    early_stopping_callback = EarlyStopping(monitor=main_metric, patience=config["patience"], mode=mode)
    checkpoint_callback = ModelCheckpoint(filename="best", monitor=main_metric, mode=mode)

    # --- Trainer ---
    L.seed_everything(config["seed"] + boost_round)
    pipeline = DlPipeline(config)

    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = [0]
    else:
        accelerator = "cpu"
        devices = 1

    trainer = L.Trainer(
        accelerator=accelerator, devices=devices,
        max_epochs=config["epochs"],
        logger=logger,
        callbacks=[early_stopping_callback, checkpoint_callback]
    )
    trainer.fit(pipeline, dm)

    # --- Load best model and get performance ---
    best_model_path = checkpoint_callback.best_model_path
    if not best_model_path:
        print(f"Warning: No best model path found for round {boost_round}. Skipping.")
        return None, None, None

    print(f"Best model for round {boost_round}: {best_model_path}")
    best_pipeline = DlPipeline.load_from_checkpoint(best_model_path, config=config)

    # Get validation performance for selecting the top models later
    val_results = trainer.validate(best_pipeline, dm)
    val_performance = val_results[0][f'best_{main_metric}']

    # Return the trained model, its performance, and the trainer instance for later use
    return best_pipeline, val_performance, trainer, dm, best_model_path


def run_boosting_experiment(config):
    """
    Main controller for the AdaBoost-style training process.
    """
    # Initialize uniform sample weights
    # We need to know the size of the training set first
    temp_dm = EhrDataModule(f'my_datasets/ehr/{config["dataset"]}/processed/fold_{config["fold"]}', task=config["task"], batch_size=config["batch_size"])
    temp_dm.setup('fit')
    n_train_samples = len(temp_dm.train_dataset)
    sample_weights = torch.full((n_train_samples,), 1.0 / n_train_samples)

    # Store results of each round
    ensemble_members = []

    # --- Boosting Loop ---
    for i in range(config['n_estimators']):
        # 1. Train a model with current sample weights
        pipeline, val_perf, trainer, dm, model_path = run_single_boost_round(config, boost_round=i, sample_weights=sample_weights)

        if pipeline is None:
            continue

        # Store the member's info
        ensemble_members.append({
            "model_path": model_path,
            "val_performance": val_perf,
            "round": i
        })

        # 2. Update weights based on training set errors (AdaBoost.M1 algorithm)
        print(f"Updating weights based on model from round {i}...")

        # Get predictions on the full, unshuffled training set
        train_loader_for_update = dm.train_dataloader_for_update()

        # Use trainer.predict to get raw outputs
        raw_preds = trainer.predict(pipeline, train_loader_for_update)
        preds_tensor = torch.cat([p[0] for p in raw_preds])
        labels_tensor = torch.cat([p[1][:, -1].unsqueeze(-1) for p in raw_preds])

        # Ensure tensors are on CPU for numpy conversion
        preds_tensor = preds_tensor.cpu().squeeze()
        labels_tensor = labels_tensor.cpu().squeeze()

        # For classification tasks
        if config["task"] in ["mortality", "readmission", "sptb"]:
            # Convert logits to predictions (0 or 1)
            predictions = (preds_tensor > 0.5).float()

            # Identify misclassified samples
            is_wrong = (predictions != labels_tensor).float()

            # Calculate weighted error
            error = torch.sum(sample_weights * is_wrong) / torch.sum(sample_weights)

            if error >= 0.5:
                print(f"Warning: Round {i} model error is {error:.4f} >= 0.5")

            if error <= 1e-10:
                print(f"Model has near-zero error {error:.4f} on training set")

            # Calculate model importance (alpha)
            alpha = 0.5 * torch.log((1 - error + 1e-10) / (error + 1e-10))

            # Update weights: increase weight of misclassified samples
            new_weights = sample_weights * torch.exp(alpha * is_wrong)

            # Normalize weights
            sample_weights = new_weights / torch.sum(new_weights)
            print(f"Updated weights: {sample_weights}")
        else:
            # For regression (e.g., 'los'), AdaBoost.R2 is more complex.
            # We will stop here and print a message.
            print("AdaBoost weight update for regression is not implemented. Stopping after one round.")
            break

    # --- Select and Save Top 3 Models ---
    if not ensemble_members:
        print("No models were successfully trained. Exiting.")
        return

    print("\nBoosting complete. Selecting top 3 models based on validation performance.")

    # Sort models by validation performance
    sort_reverse = True if config["main_metric"] == "auroc" else False
    sorted_members = sorted(ensemble_members, key=lambda x: x['val_performance'], reverse=sort_reverse)

    top_3_members = sorted_members[:3]

    # Create a directory to save the top 3 checkpoints
    save_dir = os.path.join(config["output_root"], f"{config['dataset']}/{config['task']}/{config['model']}", "top_3_boosted_models")
    os.makedirs(save_dir, exist_ok=True)

    print(f"Saving top 3 models to: {save_dir}")

    performance_summary = []
    temp_dm.setup('test')

    for i, member in enumerate(top_3_members):
        rank = i + 1
        original_path = member['model_path']
        dest_filename = f"rank_{rank}_round_{member['round']}_val_{config['main_metric']}_{member['val_performance']:.4f}.ckpt"
        dest_path = os.path.join(save_dir, dest_filename)

        print(f"  - Rank {rank}: Copying '{original_path}' to '{dest_path}'")
        shutil.copy(original_path, dest_path)

        pipeline = DlPipeline.load_from_checkpoint(dest_path, config=config)
        trainer.test(pipeline, temp_dm)
        perf = pipeline.test_performance
        outs = pipeline.test_outputs
        pd.to_pickle(outs, os.path.join(save_dir, f"rank_{rank}_round_{member['round']}_test_outs.pkl"))

        # Also save performance for easy reference
        performance_summary.append({
            "rank": rank,
            "round": member['round'],
            f"val_{config['main_metric']}": member['val_performance'],
            f"test_{config['main_metric']}": perf[config['main_metric']],
            "checkpoint_path": dest_path
        })

    perf_df = pd.DataFrame(performance_summary)
    perf_df.to_csv(os.path.join(save_dir, "top_3_summary.csv"), index=False)

    print("\nTop 3 models saved successfully.")


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate deep learning models for EHR data")
    parser.add_argument("--model", "-m", type=str, default="ConCare", help="Model name")
    parser.add_argument("--dataset", "-d", type=str, default="esrd", help="Dataset name", choices=["cdsl", "mimic-iv", "esrd", "obstetrics"])
    parser.add_argument("--task", "-t", type=str, default="mortality", help="Task name", choices=["mortality", "readmission", "sptb"])
    parser.add_argument("--hidden_dim", "-hd", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", "-bs", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", "-e", type=int, default=30, help="Number of epochs")
    parser.add_argument("--patience", "-p", type=int, default=10, help="Patience for early stopping")
    parser.add_argument("--output_dim", "-od", type=int, default=1, help="Output dimension")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--n_estimators", type=int, default=10, help="Number of boosting rounds (models to train)")
    parser.add_argument("--output_root", type=str, default="logs", help="Root directory for saving outputs")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()

    model_name = args.model

    config = {
        "model": model_name,
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
        "output_root": args.output_root,
        "main_metric": "auroc",
        "fold": 0
    }

    dataset_path = f'my_datasets/ehr/{config["dataset"]}/processed/fold_{config["fold"]}'
    config["los_info"] = None

    if args.dataset == "mimic-iv":
        config["demo_dim"] = 2
        config["lab_dim"] = 42
    elif args.dataset == "esrd":
        config["demo_dim"] = 0
        config["lab_dim"] = 17
    elif args.dataset == "obstetrics":
        config["demo_dim"] = 0
        config["lab_dim"] = 32
        config["split"] = "solo"
    elif args.dataset == "cdsl":
        config["demo_dim"] = 2
        config["lab_dim"] = 97
    else:
        raise ValueError("Unsupported dataset.")

    # Run the main boosting experiment
    run_boosting_experiment(config)

    print("\nAll experiments completed.")