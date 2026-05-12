import subprocess


experiments = [
    # # ============================================================
    # # Phase 1: extend the current best setting
    # # Current best:
    # #   noaug, lr=5e-5, batch_size=2, epochs=20
    # # Goal:
    # #   Check whether the model is still improving after 20 epochs.
    # # ============================================================
    # {
    #     "run_name": "unet_isic_noaug_lr5e5_e40",
    #     "augmentation_type": "noaug",
    #     "epochs": 40,
    #     "batch_size": 2,
    #     "lr": 5e-5,
    # },

    # # ============================================================
    # # Phase 2: augmentation comparison at the best LR
    # # Goal:
    # #   Is augmentation actually helping, or is ISIC small enough
    # #   that noaug performs better under this architecture?
    # # ============================================================
    # {
    #     "run_name": "unet_isic_geomaug_lr5e5_e40",
    #     "augmentation_type": "geomaug",
    #     "epochs": 40,
    #     "batch_size": 2,
    #     "lr": 5e-5,
    # },
    # {
    #     "run_name": "unet_isic_mildaug_lr5e5_e40",
    #     "augmentation_type": "mildaug",
    #     "epochs": 40,
    #     "batch_size": 2,
    #     "lr": 5e-5,
    # },
    # {
    #     "run_name": "unet_isic_strongaug_lr5e5_e40",
    #     "augmentation_type": "strongaug",
    #     "epochs": 40,
    #     "batch_size": 2,
    #     "lr": 5e-5,
    # },

    # # ============================================================
    # # Phase 3: LR search around the current best
    # # Goal:
    # #   1e-4 was decent, 5e-5 was better.
    # #   Try lower values to see if training becomes more stable.
    # # ============================================================
    # {
    #     "run_name": "unet_isic_noaug_lr3e5_e40",
    #     "augmentation_type": "noaug",
    #     "epochs": 40,
    #     "batch_size": 2,
    #     "lr": 3e-5,
    # },
    # {
    #     "run_name": "unet_isic_mildaug_lr3e5_e40",
    #     "augmentation_type": "mildaug",
    #     "epochs": 40,
    #     "batch_size": 2,
    #     "lr": 3e-5,
    # },
    # {
    #     "run_name": "unet_isic_noaug_lr7e5_e40",
    #     "augmentation_type": "noaug",
    #     "epochs": 40,
    #     "batch_size": 2,
    #     "lr": 7e-5,
    # },
    {
        "run_name": "unet_isic_mildaug_lr7e5_e40",
        "augmentation_type": "mildaug",
        "epochs": 40,
        "batch_size": 2,
        "lr": 7e-5,
    },

    # ============================================================
    # Phase 4: longer best-candidate runs
    # Goal:
    #   Run only the most likely good configs longer.
    #   These are good candidates based on your current results.
    # ============================================================
    # {
    #     "run_name": "unet_isic_noaug_lr5e5_e60",
    #     "augmentation_type": "noaug",
    #     "epochs": 60,
    #     "batch_size": 2,
    #     "lr": 5e-5,
    # },
    # {
    #     "run_name": "unet_isic_mildaug_lr5e5_e60",
    #     "augmentation_type": "mildaug",
    #     "epochs": 60,
    #     "batch_size": 2,
    #     "lr": 5e-5,
    # },
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