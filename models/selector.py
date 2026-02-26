import torch 

from .rnn_only import RNNOnly
from .cnn_only import CNNOnly
from .cnn_and_rnn import CNNAndRNN
from .cnn_and_rnn_and_funnel import CNNAndRNNAndFunnel
from .big_rnn_and_funnel import BigRNNAndFunnel
from .v2 import V2
from .v2_funnel import V2Funnel
from .v2_small import V2Small
from .v2_large import V2Large
from .v2_extra_large import V2ExtraLarge 
from .v2_coolio import V2Coolio
from .cr import CR
from .crs import CRS
from .phase_tcn import PhaseTCN
from .phase_tcn_mel import PhaseTCNMel
from .phase_lstm_mel import PhaseLSTMMel
from .phase_gru_mel import PhaseGRUMel
from .phase_gru import PhaseGRU

models = {
    'cnn_only': CNNOnly,
    'rnn_only': RNNOnly,
    'cnn_and_rnn': CNNAndRNN,
    'cnn_and_rnn_and_funnel': CNNAndRNNAndFunnel,
    'big_rnn_and_funnel': BigRNNAndFunnel,
    'v2': V2,
    'v2_funnel': V2Funnel, 
    'v2_small': V2Small,
    'v2_large': V2Large, 
    'v2_extra_large': V2ExtraLarge, 
    'v2_coolio': V2Coolio, 
    'CR': CR,
    'CRS': CRS,
    'phase_tcn': PhaseTCN,
    'phase_tcn_mel': PhaseTCNMel,
    'phase_lstm_mel': PhaseLSTMMel,
    'phase_gru_mel': PhaseGRUMel,
    'phase_gru': PhaseGRU,
}

def getModels():
    return list(models.keys())

class ModelNotImplemented(Exception):
    pass

def getModelClass(name):
    for key, value in models.items():
        if key == name:
            return value
    raise ModelNotImplemented('invalid model')

def loadModel(file):
    obj = torch.load(file)

    model_type = obj['model_type']
    modelClass = getModelClass(model_type)

    print('loading model: ', model_type)

    model_hparams = obj.get('model_hparams', {})
    try:
        model = modelClass(**model_hparams)
    except TypeError:
        model = modelClass()
    model.load_state_dict(obj['model_state_dict'])

    return model, obj

def saveModel(file, model, obj):
    for model_type, model_class in models.items():
        if isinstance(model, model_class):
            save_obj = {
                'model_type': model_type,
                'model_state_dict': model.state_dict(),
            }
            if hasattr(model, 'hparams'):
                save_obj['model_hparams'] = model.hparams
            torch.save({**save_obj, **obj}, file)
