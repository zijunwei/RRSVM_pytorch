from torch.nn import Parameter

def load_state_dict(model, state_dict):
    """Copies parameters and buffers from :attr:`state_dict` into
    this module and its descendants. The keys of :attr:`state_dict` must
    exactly match the keys returned by this module's :func:`state_dict()`
    function.

    Arguments:
        state_dict (dict): A dict containing parameters and
            persistent buffers.
    """
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name not in own_state:
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        try:
            own_state[name].copy_(param)
        except:
            print('While copying the parameter named {}, whose dimensions in the model are'
                  ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                name, own_state[name].size(), param.size()))
            raise

    missing = list(set(own_state.keys()) - set(state_dict.keys()))
    missing = sorted(missing)
    if len(missing) > 0:
        for s_missing in missing:
            if s_missing[-1] == 's':
                # print "{:s} is NOT loaded from Orig Model".format(s_missing)
                continue
            else:
                raise KeyError('missing keys in state_dict: "{}"'.format(s_missing))