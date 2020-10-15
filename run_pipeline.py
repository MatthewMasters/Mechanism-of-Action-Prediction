from pipeline.utils import seed_everything, Config, SETTINGS
from pipeline.engine import Engine

seed_everything()

# Read config
config = Config('configs/MoA_baseline_002.py')
# config = Config('configs/MoA_feature_selection.py')

# Setup model engine
engine = Engine(config)
engine.run()
