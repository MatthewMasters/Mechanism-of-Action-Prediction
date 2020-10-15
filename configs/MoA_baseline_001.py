

config = dict(
    # Project settings
    project_name='MoA_baseline_001',

    # Validation
    n_folds=5,

    # Preprocessing
    outlier_removal=True,  # TODO

    # General training

    # Ensemble of models
    ensemble=[
        dict(
            # Model
            type='NN',
            model='MoaDenseNet',
            n_hidden_layer=1,
            dropout=0.5,
            hidden_dim=1024,
            activation='relu',  # prelu, relu
            normalization='batch',

            # Training
            batch_size=512,
            num_workers=4,
            n_epochs=25,
            optimizer='adam',
            learning_rate=0.001,
            weight_decay=0.00001,
            scheduler='OneCycleLR',
            use_smote=False,
            use_amp=False,
            verbose=True,

            # Augmentation (noise, mixup, cutout, cutmix)
            augmentations=[]
        ),
    ],

    # Ensembling
    oom_method='blend',

    # Postprocessing
    surety=True,  # TODO
    smoothing=True,  # TODO
)
