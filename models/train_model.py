from DTAN.alignment_loss import alignment_loss
from tqdm import tqdm
import torch.optim as optim
import torch
from DTAN.DTAN_layer import DTAN
import numpy as np


def train(train_loader, val_loader, DTANargs, Experiment, print_model=False):
    """

    Args:
        train_loader: PyTorch data loader for iterating over the train set
        val_loader: PyTorch data loader for iterating over the validation set
        DTANargs: DTAN args class, defined at train_utils.py
        Experiment: Experiemnts class degined at train_utils.py
        print_model: bool - print DTAN details (#params, architecture, CPAB basis

    Returns:
        trained DTAN model (nn.module)

    """

    # Init DTAN class
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    channels, input_shape = train_loader.dataset[0][0].shape

    model = DTAN(input_shape, channels, tess=[DTANargs.tess_size,], n_recurrence=DTANargs.n_recurrences,
                    zero_boundary=DTANargs.zero_boundary, device='gpu').to(device)

    DTANargs.T = model.get_basis()
    optimizer = optim.Adam(model.parameters(), lr=Experiment.lr)


    # Print model
    if print_model:
        print(model)
        print(DTANargs)
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("# parameters:", pytorch_total_params)

    # Train
    min_loss = np.inf
    for epoch in tqdm(range(1, Experiment.n_epochs+1)):
        train_loss = train_epoch(train_loader, device, optimizer, model, channels, DTANargs)
        val_loss = validation_epoch(val_loader, device, model, channels,DTANargs)
        # save checkpoint
        if val_loss < min_loss:
            min_loss = val_loss
            _save_checkpoint(model, optimizer, val_loss, Experiment.exp_name)
        if epoch % 50 == 0:
            train_loss /= len(train_loader.dataset)
            print('\nTrain set: Average loss: {:.4f}\n'.format(train_loss))
            val_loss /= len(val_loader.dataset)
            print('Validation set: Average loss: {:.4f}\n'.format(val_loss))


    # Load best model based on validation loss
    checkpoint = torch.load(f'../checkpoints/{Experiment.exp_name}_checkpoint.pth')


    return model


def train_epoch(train_loader, device, optimizer, model, channels, DTANargs):
    """

    Args:
        train_loader: PyTorch data loader for iterating over the train set
        device: device used by the network: 'cuda', 'cpu', 'gpu' (str)
        optimizer: PyTorch optimizer class
        model: DTAN model to train (nn.module)
        channels: number of input channels (int)
        DTANargs: DTAN args class, defined at train_utils.py

    Returns:
        train loss (float)

    """
    model.train()
    # CPAB basis, used for smoothness prior computation
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output, thetas = model(data, return_theta=True)

        loss = alignment_loss(output, target, thetas, channels, DTANargs)
        loss.backward()
        optimizer.step()

        return loss


def validation_epoch(val_loader, device, model, channels, DTANargs):
    with torch.no_grad():
        model.eval()
        val_loss = 0
        prior_loss = 0
        align_loss = 0
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output, theta = model(data, return_theta=True)

            # sum up batch loss
            val_loss += alignment_loss(output, target, theta, channels, DTANargs)
            # get the index of the max log-probability

        return val_loss


def test(epoch, test_loader, device, optimizer, model, min_loss, DTANargs):
    with torch.no_grad():
        model.eval()
        test_loss = 0
        prior_loss = 0
        align_loss = 0
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, theta = model(data, return_theta=True)
            test_loss += alignment_loss(output, target, theta, DTANargs)



def _save_checkpoint(model, optimizer, test_loss, exp_name=''):
    """

    Args:
        model: DTAN model (nn.module)
        optimizer: PyTorch optimizer class
        test_loss: float, test loss at time of saving the checkpoint
        exp_name: file name (str)
    """
    #print("saving model checkpoint")
    checkpoint = {'model_state_dict': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'loss': test_loss
                  }

    torch.save(checkpoint, f'../checkpoints/{exp_name}_checkpoint.pth')