import numpy as np

class ExperimentClass():
    def __init__(self, n_epochs, batch_size, lr, exp_name, device="cpu"):
        # Training
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.exp_name = exp_name
        self.device = device
        self.dtan_network = None

        self.loss_tracker = {
            "train": self.loss_dict(name="train"),
            "validation": self.loss_dict(name="validation"),
            "test": self.loss_dict(name="test")
        }

    def add_DTAN_arg(self, DTAN_args):
        self.DTAN_args = DTAN_args

    def get_DTAN_args(self):
        return self.DTAN_args

    def add_model(self, model):
        self.model = model

    def get_model(self):
        return self.model

    def update_loss(self, loss_type, epoch, alignment_loss, prior_loss):
        total_loss = alignment_loss + prior_loss
        self.loss_tracker[loss_type]["total_loss"].append(total_loss)
        self.loss_tracker[loss_type]["alignment_loss"].append(alignment_loss)
        self.loss_tracker[loss_type]["prior_loss"].append(prior_loss)
        # in case update loss is called every X epochs
        self.loss_tracker[loss_type]["epoch"].append(epoch)

    def get_min_loss(self, train_type="train", loss_type="total_loss",return_epoch = False):
        loss_arr = np.asarray(self.loss_tracker[train_type][loss_type])
        min_loss =loss_arr.min()
        if not return_epoch:
           return min_loss
        else:
            epoch_arr = np.asarray(self.loss_tracker[train_type][loss_type])
            min_loss_idx = loss_arr.argmin()
            epoch = epoch_arr[min_loss_idx]
            return min_loss, epoch

    def print_min_loss_all(self):
        print("--- Printing minimum loss --")
        for train_type in ["train", "validation", "test"]:
            total_loss, min_epoch = self.get_min_loss(train_type, "total_loss", True)
            print(f"{train_type} minimum (total) loss: {total_loss}")

    def loss_dict(self, name):
        loss_dict = {
            "train_type": name,
            "total_loss": [],
            "alignment_loss": [],
            "smoothness_pior_loss": [],
            "epoch": [],
        }
        return loss_dict

    def add_DTAN_model(self, dtan_net):
        self.dtan_network = dtan_net

    def get_DTAN_model(self):
        return self.dtan_network

    def __str__(self):
        return str(self.__dict__)

class ExperimentsManager():
    def __init__(self):
        self.experiments_dict = {}

    def add_experiment(self, exp_name, n_epochs, batch_size, lr, device):
        self.experiments_dict[exp_name] = ExperimentClass(
            n_epochs, batch_size, lr, exp_name, device
        )

    def get_experiment(self, exp_name):
        return self.__getitem__(exp_name)

    def __getitem__(self, exp_name):
        return self.experiments_dict[exp_name]


    def __str__(self):
        return str(self.__dict__)


class DTAN_args:
    def __init__(self,
                 tess_size = 32, smoothness_prior = True, lambda_smooth = 1,
                 lambda_var = 0.1, n_recurrences = 1, zero_boundary = True, T=None
                 ):
        self.tess_size = tess_size
        self.tess_size = tess_size
        self.zero_boundary = zero_boundary
        self.smoothness_prior = smoothness_prior
        self.lambda_smooth = lambda_smooth
        self.lambda_var = lambda_var
        self.n_recurrences = n_recurrences
        self.T = T


    def __str__(self):
        return str(self.__dict__)

    def set_basis(self, T):
        self.T = T
