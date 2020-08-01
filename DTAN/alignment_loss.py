import tensorflow as tf
from DTAN.smoothness_prior import smoothness_norm


def alignment_loss(X_trasformed, labels, thetas, n_channels, DTANargs):
    '''
    Torch data format is  [N, C, W] W=timesteps
    Args:
        X_trasformed:
        labels:
        thetas:
        DTANargs:

    Returns:

    '''
    loss = 0
    align_loss = 0
    prior_loss = 0
    n_classes = labels.unique()
    for i in n_classes:
        X_within_class = X_trasformed[labels==i]
        if n_channels == 1:
            # Single channel variance across samples
            loss += X_within_class.var(dim=0, unbiased=False).mean()
        else:
            # variance between signalls in each channel (dim=0)
            # mean over each channel (dim=1)
            per_channel_loss = X_within_class.var(dim=0, unbiased=False).mean(dim=1)
            per_channel_loss = per_channel_loss.mean()
            loss += per_channel_loss

    loss /= len(n_classes)
    #print("\nDEBUG Alignment loss:", loss.item())
    if DTANargs.smoothness_prior:
        for theta in thetas:
            # alignment loss takes over variance loss
            # larger penalty when k increases -> coarse to fine
            # TODO not tested with several channels
            prior_loss += 0.1*smoothness_norm(DTANargs.T, theta, DTANargs.lambda_smooth, DTANargs.lambda_var, print_info=False)
        loss += prior_loss
        #print("DEBUG Prior loss:", prior_loss.item())
    return loss
