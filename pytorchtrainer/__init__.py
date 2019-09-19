
__version__ = '0.2.0rc3'


from .trainer import create_default_trainer, ModuleTrainer, State
from . import callback
from . import metric
from . import stop_condition
