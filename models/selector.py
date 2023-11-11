import torch 

from .rnn_only import RNNOnly
from .cnn_only import CNNOnly
from .cnn_and_rnn import CNNAndRNN
from .cnn_and_rnn_and_funnel import CNNAndRNNAndFunnel
from .big_rnn_and_funnel import BigRNNAndFunnel
from .rnn_and_2_funnels import RNNAnd2Funnels
from .separate_lanes import SeparateLanes
from .v2 import V2

models = {
    'cnn_only': CNNOnly,
    'rnn_only': RNNOnly,
    'cnn_and_rnn': CNNAndRNN,
    'cnn_and_rnn_and_funnel': CNNAndRNNAndFunnel,
    'big_rnn_and_funnel': BigRNNAndFunnel,
    'rnn_and_2_funnels': RNNAnd2Funnels,
    'separate_lanes': SeparateLanes, 
    'rnn_2f_dropout': RNNAnd2Funnels,
    'v2': V2
}

def getModels():
    return list(models.keys())

def getModelClass(name): 
    for key in models.keys():
        if key == name:
            return models[key]
    raise Exception('invalid model')

def loadModel(file):
    obj = torch.load(file)

    model_type = obj['model_type']
    modelClass = getModelClass(model_type)

    model = modelClass()
    model.load_state_dict(obj['model_state_dict']) 

    return model, obj 

def saveModel(file, model, obj):
    for model_type, model_class in models.items(): 
        if isinstance(model, model_class):
            torch.save({
                'model_type': model_type,
                'model_state_dict': model.state_dict(),
                **obj
            }, file)

