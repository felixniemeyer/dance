from .rnn_only import RNNOnly
from .cnn_only import CNNOnly
from .cnn_and_rnn import CNNAndRNN
from .cnn_and_rnn_and_funnel import CNNAndRNNAndFunnel
from .big_rnn_and_funnel import BigRNNAndFunnel
from .rnn_and_2_funnels import RNNAnd2Funnels
from .separate_lanes import SeparateLanes

models = {
    'cnn_only': CNNOnly,
    'rnn_only': RNNOnly,
    'cnn_and_rnn': CNNAndRNN,
    'cnn_and_rnn_and_funnel': CNNAndRNNAndFunnel,
    'big_rnn_and_funnel': BigRNNAndFunnel,
    'rnn_and_2_funnels': RNNAnd2Funnels,
    'separate_lanes': SeparateLanes
}

def getModels():
    return list(models.keys())

def getModelClass(name): 
    for key in models.keys():
        if key == name:
            return models[key]
    print('valid models are:')
    for key in models.keys():
        print(key)
    raise Exception('invalid model')
