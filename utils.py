import torch


def save_checkpoint(state, filename):
    """ saving model's weights """
    print ('=> saving checkpoint')
    torch.save(state, filename)


def load_checkpoint(checkpoint_path, model):
    """ loading model's weights """
    print ('=> loading checkpoint')
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])


def str2bool(v):
    """argparse handels type=bool in a weird way.
    See this stack overflow: https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    we can use this function as type converter for boolean values
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')