

config = dict(
    # Project settings
    project_name='MoA_feature_selection',
    batch_size=1024,
    num_workers=4,

    # Validation
    n_folds=5,

    # Preprocessing
    outlier_removal=True,  # TODO

    # General training

    # Ensemble of models
    ensemble=[
        dict(
            # Model
            type='Target',

            # Feature selection
            n_features=20,
            n_cutoff=20,  # number of instances required, otherwise set prediction to 0

            model=dict(
                model='MoaDenseNet',
                n_hidden_layer=2,
                dropout=0.5,
                hidden_dim=32,
                activation='prelu',  # prelu, relu
                normalization='batch',

                # Training
                batch_size=512,
                num_workers=4,
                n_epochs=25,
                optimizer='adam',
                learning_rate=0.001,
                weight_decay=0.00001,
                scheduler='OneCycleLR',
                use_smart_init=True,
                use_smote=False,
                use_amp=False,
                verbose=True,

                # Augmentation
                augmentations=[],
            )
        ),
    ],

    # Ensembling
    ensemble_method='mean',  # TODO

    # Postprocessing
    surety=True,  # TODO
)
