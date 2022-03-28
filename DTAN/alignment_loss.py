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
            # variance between signals in each channel (dim=1)
            # mean variance of all channels and samples (dim=0)
            per_channel_loss = X_within_class.var(dim=1, unbiased=False).mean(dim=0)
            per_channel_loss = per_channel_loss.mean()
            loss += per_channel_loss

    loss /= len(n_classes)
    # Note: for multi-channel data, assues same transformation (i.e., theta) for all channels
    if DTANargs.smoothness_prior:
        for theta in thetas:
            # alignment loss takes over variance loss
            # larger penalty when k increases -> coarse to fine
            prior_loss += 0.1*smoothness_norm(DTANargs.T, theta, DTANargs.lambda_smooth, DTANargs.lambda_var, print_info=False)
        loss += prior_loss
    return loss
