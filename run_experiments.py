import subprocess


experiments = [
    # ------------------------------------------------------------
    # Main continuation of current best setting
    # ------------------------------------------------------------
    {
        "run_name": "unet_isic_noaug_lr5e5_e40",
        "batch_size": 2,
        "epochs": 40,
        "lr": 5e-5,
        "augmentation_type": "noaug",
    },
    {
        "run_name": "unet_isic_lightaug_lr5e5_e40",
        "batch_size": 2,
        "epochs": 40,
        "lr": 5e-5,
        "augmentation_type": "lightaug",
    },

    # ------------------------------------------------------------
    # Learning-rate sweep around current best
    # ------------------------------------------------------------
    {
        "run_name": "unet_isic_noaug_lr7e5_e40",
        "batch_size": 2,
        "epochs": 40,
        "lr": 7e-5,
        "augmentation_type": "noaug",
    },
    {
        "run_name": "unet_isic_noaug_lr3e5_e40",
        "batch_size": 2,
        "epochs": 40,
        "lr": 3e-5,
        "augmentation_type": "noaug",
    },
    {
        "run_name": "unet_isic_noaug_lr1e5_e40",
        "batch_size": 2,
        "epochs": 40,
        "lr": 1e-5,
        "augmentation_type": "noaug",
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