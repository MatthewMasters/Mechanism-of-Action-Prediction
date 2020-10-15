

config = dict(
    # Project settings
    project_name='MoA_baseline_001',
    batch_size=1024,
    num_workers=4,

    # Validation
    n_folds=5,

    # Preprocessing
    outlier_removal=True,  # TODO

    # General training

    # Ensemble of models
    ensemble=[
        # dict(
        #     # Model
        #     type='NN',
        #     model='MoaDenseNet',
        #     feature_set='all',
        #     target_set='all',
        #     n_hidden_layer=1,
        #     dropout=0.5,
        #     hidden_dim=512,
        #     activation='prelu',  # prelu, relu
        #     normalization='batch',
        #
        #     # Training
        #     batch_size=512,
        #     num_workers=4,
        #     n_epochs=25,
        #     optimizer='adam',
        #     learning_rate=0.001,
        #     weight_decay=0.00001,
        #     scheduler='OneCycleLR',
        #     use_amp=False,
        #     verbose=True,
        #
        #     # Augmentation
        #     normal_noise=True,  # TODO
        # ),
        dict(
            type='kNN',
            clip_threshold=None,
            amplify=None,
            max_distance=300,
        )
    ],

    # Ensembling
    ensemble_method='mean',  # TODO

    # Postprocessing
    surety=True,  # TODO
)
