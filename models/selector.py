import torch

from .phase_gru_mel import PhaseGRUMel
from .phase_gru import PhaseGRU
from .phase_lstm_mel import PhaseLSTMMel

models = {
    'phase_gru_mel':  PhaseGRUMel,
    'phase_gru':      PhaseGRU,
    'phase_lstm_mel': PhaseLSTMMel,
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

def loadModel(file, map_location=None):
    if map_location is None and not torch.cuda.is_available():
        map_location = torch.device('cpu')
    obj = torch.load(file, map_location=map_location)

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
