# Args class for running DTAN - makes life easier

class args:
    def __init__(self, tess_size = 32, smoothness_prior = True, lambda_smooth = 1,
                 lambda_var = 0.1, n_recurrences = 1, zero_boundary = True):
        self.tess_size = tess_size
        self.zero_boundary = zero_boundary
        self.smoothness_prior = smoothness_prior
        self.lambda_smooth = lambda_smooth
        self.lambda_var = lambda_var
        self.n_recurrences = n_recurrences

        # Training
        self.n_epochs = 2000

    def __str__(self):
        return str(self.__dict__)