def N_log_prob(mu, log_var, y):
    """
    Evaluate the negative log probability -log p( y | z ) = -log N( y | mu(z), sigma(z) )
    of the independent Guassian likelihood for observed data y.
    mu: N x T x P
    log_var: N x T x P
    y: N x T x P
    log_prob: 1
    """

    # norm = np.log( 2 * np.pi ) + log_var # ignore normalozation factor
    # norm =  log_var
    # square = (y - mu).pow(2) / log_var.exp()

    # neg_log_prob = 0.5*(norm + square)#.sum()
    neg_log_prob = log_var + (y - mu).pow(2) / log_var.exp()

    return neg_log_prob
