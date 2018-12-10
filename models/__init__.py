

#from .enet import *


def get_segmentation_model(name, **kwargs):
    from .enet import get_enet

    models = {
        'enet': get_enet,

    }
    return models[name.lower()](**kwargs)
