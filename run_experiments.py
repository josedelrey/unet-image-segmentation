import subprocess


experiments = [
    {
        "run_name": "unet_original_isic_noaug_lr1e4_e20",
        "batch_size": 2,
        "epochs": 20,
        "lr": 1e-4,
        "augmentation_type": "noaug",
    },
    {
        "run_name": "unet_original_isic_noaug_lr3e4_e20",
        "batch_size": 2,
        "epochs": 20,
        "lr": 3e-4,
        "augmentation_type": "noaug",
    },
    {
        "run_name": "unet_original_isic_noaug_lr1e3_e20",
        "batch_size": 2,
        "epochs": 20,
        "lr": 1e-3,
        "augmentation_type": "noaug",
    },
    {
        "run_name": "unet_original_isic_lightaug_lr_best_e20",
        "batch_size": 2,
        "epochs": 20,
        "lr": 1e-4,  # change manually after seeing best LR
        "augmentation_type": "lightaug",
    },
    {
        "run_name": "unet_original_isic_best_e50",
        "batch_size": 2,
        "epochs": 50,
        "lr": 1e-4,  # change manually after seeing best LR
        "augmentation_type": "lightaug",  # change to noaug if noaug was better
    },
]


for exp in experiments:
    command = [
        "python",
        "src/train.py",
        "--run_name", exp["run_name"],
        "--batch_size", str(exp["batch_size"]),
        "--epochs", str(exp["epochs"]),
        "--lr", str(exp["lr"]),
        "--augmentation_type", exp["augmentation_type"],
    ]

    print("\n" + "=" * 80)
    print("Running experiment:", exp["run_name"])
    print("=" * 80)

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"FAILED: {exp['run_name']} with return code {e.returncode}")
        print("Continuing with next experiment...")