
__version__ = '0.1.0-rc1'


from .trainer import create_default_trainer, ModuleTrainer, State
from . import callback
from . import metric
from . import stop_condition
