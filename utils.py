import typing as tp
import os
import torch
import numpy as np

import swyft
import pytorch_lightning as pl 

 
def get_store(store_path, sim: tp.Optional[swyft.Simulator] = None, N_sims: tp.Optional[int] = 50_000, chunk_size: int = 1024, batch_size: int = 1024) -> swyft.ZarrStore:
    """
    Get swyft Zarr store. 
    If sim is not None, simulate N_sims and store them in the store.
    """ 
    store = swyft.ZarrStore(store_path)
    if sim is not None:
        shapes, dtypes = sim.get_shapes_and_dtypes()
        store.init(N_sims, chunk_size, shapes, dtypes)
        store.simulate(sim.sample, batch_size=batch_size)

    return store
    

def get_trainer(trainer_name:tp.Optional[str] = "task", save_dir:tp.Optional[str] = "./lightning_logs", patience:int=6, max_epochs:int=100):
    """
    Instantiate swyft trainer with callbacks
    """
    
    logger = pl.loggers.TensorBoardLogger(save_dir=save_dir, name = trainer_name, version=None, sub_dir=None)
    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor = 'val_loss', save_top_k = 1), 
        pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        pl.callbacks.EarlyStopping(monitor="val_loss", patience = patience, mode='min')
    ]

    trainer = swyft.SwyftTrainer(
        accelerator = "gpu",
        devices = 1,
        logger = logger, 
        max_epochs = max_epochs,
        callbacks = callbacks,
    )
    
    return trainer, callbacks


def save_model(callbacks):
    """ Save network model and store its path."""
    callbacks[0].best_model_path
    callbacks[0].to_yaml(callbacks[0].dirpath+"/model.yaml")
    ckpt_path = swyft.best_from_yaml(callbacks[0].dirpath+"/model.yaml")
    return ckpt_path


def get_bounds(predictions, threshold: float = 1e-6, parname: str = 'z') -> dict:
    """
    Get bounds from swyft.Trainer predictions.

    Args:
        predictions (_type_): swyft.Trainer.infer(...) output
        threshold (float, optional): threshold at which to truncate the ratio estimator. Defaults to 1e-6.
        parname (string, optional): name of the parameters to get the bounds for. Defaults to 'z'.

    Returns:
        dict: dictionary that containes the bounds for each parameter.
    """
    bounds = {}
    if isinstance(predictions, dict):
        for k, v in predictions.items():
            bounds[k] = swyft.collect_rect_bounds(v, k, (v.params.size(1),), threshold = threshold)
    else:
        bounds[parname] = swyft.collect_rect_bounds(predictions, parname, (predictions.params.size(1),), threshold = threshold)
    return bounds


def file_cache(filename, fn):
    """ Store data in file."""
    if os.path.exists(filename):
        return torch.load(filename)
    else:
        obj = fn()
        torch.save(obj, filename)
        return obj
    

def bound_check(X, bounds):
    """ Utility function for nested sampling bounds checking."""
    bounds = bounds.to(X.device)
    return (X<=(bounds[..., 1])).prod(-1)*(X>=(bounds[..., 0])).prod(-1)


def linear_rescale(v, v_ranges, u_ranges):
    """Rescales a tensor in its last dimension from v_ranges to u_ranges
    
    Args:
        v: (..., N)
        v_ranges: (N, 2)
        u_ranges: (N, 2)
    
    Returns:
        u: (..., N)
    """
    device = v.device
    v_ranges = torch.tensor(v_ranges)
    u_ranges = torch.tensor(u_ranges)

    # Move points onto hypercube
    v_bias = v_ranges[:,0].to(device)
    v_width = (v_ranges[:,1]-v_ranges[:,0]).to(device)
    
    # Move points onto hypercube
    u_bias = u_ranges[:,0].to(device)
    u_width = (u_ranges[:,1]-u_ranges[:,0]).to(device)

    t = (v - v_bias)/v_width
    u = (t*u_width+u_bias)
    return u


def log_likelihood(z, net, obs, bounds_z):
    """Neural log-likelihood function for nested sampler"""

    z = linear_rescale(z, torch.tensor([0,1]).reshape(-1, 2), bounds_z.reshape(-1, 2).to(z.device)).to(z.device)
    B = swyft.Samples({'z':z})
    A = swyft.Samples({"x": obs["x"], "z": z})
    with torch.no_grad():
        predictions = net(A, B)
    if isinstance(predictions, dict):
        logl = predictions["lrs_total"].logratios.squeeze(-1)
    else:
        logl = predictions.logratios.squeeze(-1)
    return logl


class BaseTorchDataset(torch.utils.data.Dataset):
    """"Simple torch dataset class for nested sampling samples"""

    def __init__(self, root_dir, n_max = None):
        self.root_dir = root_dir
        self.X = torch.load(root_dir)
        self.n = len(self.X)
        if n_max is not None:
            self.n = min(self.n, n_max)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        z = self.X[idx]
        if idx >= self.n or idx < 0:
            raise IndexError()
        return torch.as_tensor(z)
    
    