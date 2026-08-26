import seisbench.data as sbd
import seisbench.util as sbu
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding

import numpy as np
import torch


from abc import ABC, abstractmethod

EPS = 1e-5 # To prevent division by zero

s_phase_dict = {"trace_s_arrival_sample": "S",}
p_phase_dict = {"trace_p_arrival_sample": "P",}
ps_phase_dict = {"trace_p_arrival_sample": "P", "trace_s_arrival_sample": "S",}


class ModelConfig(ABC):
    
    def get_augs(self):
        return self._augs

    @abstractmethod
    def get_new_model(self):
        """Return a fresh non fine-tuned pre-trained model from seisbench"""
        pass
    
    @abstractmethod
    def get_model_class(self):
        """Return the SeisBench model class."""
        pass
         
    def load_finetuned_model(self, model_path : str):
        model = self.get_model_class()()   # instantiate empty model
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        return model
    
    @staticmethod
    @abstractmethod    
    def loss_fn(y_pred_p_array, y_true_p_array):
        pass

class EQTransformerConfig(ModelConfig):
    
    class _MergeLabels():
        def __call__(self, sample):
            # stack box probability y_det label with gaussian y probability label
            y_det = sample["y_det"][0]
            y_ps  = sample["y"][0]
            meta = sample["y"][1]
            sample["y"] = (np.vstack([y_det, y_ps]), meta)
            return sample
        
    class _EQTDetectionLabeller(sbg.ProbabilisticLabeller):
        def label(self, X, metadata):
            length = X.shape[-1]
            p = metadata.get("trace_p_arrival_sample", None)
            s = metadata.get("trace_s_arrival_sample", None)

            if p is None or s is None or np.isnan(p) or np.isnan(s):
                y = np.zeros((1, length), dtype=np.float32)
            else:
                p = int(p)
                s = int(s)
                end = int(s + 1.4 * (s - p))
                end = min(end, length)
                y = np.zeros((1, length), dtype=np.float32)
                y[0, p:end] = 1.0

            return y
    
    def __init__(self):
        super().__init__()
        self._augs = [ # Based on EQTransformer/core/mseed_predictor.py
                sbg.Normalize(demean_axis=-1, key='X'), # Remove means channel-wise 
                sbg.Filter(N=2, Wn=[1.0, 45.0], btype='band', forward_backward=True, axis=-1, key='X'),
                CosineTaper(max_percentage=0.001, max_length=2, key="X"),
                sbg.Normalize(amp_norm_axis=-1, amp_norm_type='std', key='X'), # Std normalisaiton
                sbg.ProbabilisticLabeller(
                        label_columns=ps_phase_dict, 
                        model_labels=["P", "S"], sigma=20, dim=0, noise_column=False
                    ),  
                self._EQTDetectionLabeller(
                        label_columns=ps_phase_dict, model_labels=["P", "S"], # placeholders used for label_columns, model_labels, dim, and noise_column for Generator to work
                        dim=0, noise_column=False,
                        key=("X", "y_det")   # write into a separate key
                    ),
                self._MergeLabels(),
                sbg.ChangeDtype(np.float32, key="y")
        ]
        
    def get_new_model(self):
        return sbm.EQTransformer.from_pretrained("original")
    
    def get_model_class(self):
        return sbm.EQTransformer   
    
    @staticmethod
    def loss_fn(y_pred_p_array, y_true_p_array):
        loss = torch.nn.BCELoss()
        # Extract ground truth channels explicitly
        y_det = y_true_p_array[:, 0, :]     # (B, T)
        y_p   = y_true_p_array[:, 1, :]     # (B, T)
        y_s   = y_true_p_array[:, 2, :]     # (B, T)
        
        det_pred = y_pred_p_array[:, 0, :]  # (B, T)
        p_pred   = y_pred_p_array[:, 1, :]  # (B, T)
        s_pred   = y_pred_p_array[:, 2, :]  # (B, T)

        # Sum of three BCE losses (same as original EQTransformer)
        return (
            0.05 * loss(det_pred, y_det) +
            0.40 * loss(p_pred, y_p) +
            0.55 * loss(s_pred, y_s)
        )
        
        
class PhaseNetConfig(ModelConfig):
    def __init__(self):
        super().__init__()
        self._augs = [ 
            sbg.Normalize(demean_axis=-1, amp_norm_axis=-1, amp_norm_type='std', eps=EPS, key='X'),
            sbg.ProbabilisticLabeller(shape="gaussian", label_columns=ps_phase_dict, model_labels=["N", "P", "S"], dim=0, sigma=20),
        ]
    
    def get_new_model(self):
        return sbm.PhaseNet.from_pretrained("original")
    
    def get_model_class(self):
        return sbm.PhaseNet

    @staticmethod
    def loss_fn(y_pred_p_array, y_true_p_array):
        # vector cross entropy loss
        h = y_true_p_array * torch.log(y_pred_p_array + EPS)
        h = h.mean(-1).sum(-1)  # Mean along sample dimension and sum along pick dimension
        h = h.mean()  # Mean over batch axis
        return -h 
    

class EQCCTConfig(ModelConfig):
    def __init__(self, phase: str):
        super().__init__()

        if phase.upper() not in ["P", "S"]:
            raise TypeError(f'phase_type "{phase}" is not accepted. Use "P" or "S".')
        
        self._phase = phase.upper()

        if self._phase == "P":
            self._phase_dict = p_phase_dict
        else:
            self._phase_dict = s_phase_dict 
        
        self._augs = [
            sbg.Normalize(demean_axis=-1, key='X'), # Remove means channel-wise
            sbg.Filter(N=2, Wn=[1.0, 45.0], btype='band', forward_backward=True, axis=-1, key='X'),
            CosineTaper(max_percentage=0.001, max_length=2, key="X"),
            sbg.Normalize(amp_norm_axis=-1, amp_norm_type='std', eps=EPS, key='X'),
            sbg.ProbabilisticLabeller( # sigma=1 for almost straight vertical line at pick location
                label_columns=self._phase_dict, model_labels=[self._phase], sigma=20, dim=0, noise_column=False
            ),
            sbg.ChangeDtype(np.float32, key="y")
        ]
        
    def get_new_model(self):
        if self._phase == "P":
            return sbm.EQCCTP.from_pretrained("original")
        elif self._phase == "S":
            return sbm.EQCCTS.from_pretrained("original")
    
    def get_model_class(self):
        if self._phase == "P":
            return sbm.EQCCTP
        elif self._phase == "S":
            return sbm.EQCCTS
    
    @staticmethod
    def loss_fn(y_pred_p_array, y_true_p_array):
        # return torch.mean((y_pred_p_array - y_true_p_array)**2) # Loss function is MSE from the EQCCT paper
        loss = torch.nn.BCELoss()
        return loss(y_pred_p_array, y_true_p_array) # BCE Loss inferred from EQCCT github 
    
# Before implementing new Config classes, need to check if the number of input samples matches that of the dataset (aka 6000 samples)


class CosineTaper:
    """
    Exact ObsPy-style cosine taper:
    Equivalent to st.taper(max_percentage, type='cosine', max_length)
    """
    def __init__(self, max_percentage=0.001, max_length=2, key="X"):
        self.max_percentage = max_percentage
        self.max_length = max_length

        if isinstance(key, str):
            self.key = (key, key)
        else:
            self.key = key

    def __call__(self, state_dict):
        x, metadata = state_dict[self.key[0]]

        if isinstance(x, list):
            x = [self._taper(arr) for arr in x]
        else:
            x = self._taper(x)

        state_dict[self.key[1]] = (x, metadata)

    def _taper(self, x):
        # Convert numpy → torch
        orig_numpy = isinstance(x, np.ndarray)
        x_t = torch.tensor(x, dtype=torch.float32) if orig_numpy else x

        n_samples = x_t.shape[-1]

        # ObsPy taper length
        L = min(int(self.max_percentage * n_samples), self.max_length)
        if L <= 0:
            return x  # no taper

        # ObsPy cosine taper: w[n] = 0.5 * (1 - cos(pi*n/L))
        n = torch.arange(L, device=x_t.device)
        w = 0.5 * (1 - torch.cos(np.pi * n / L))

        # Apply taper
        x_t[:, :L] *= w
        x_t[:, -L:] *= torch.flip(w, dims=(0,))

        return x_t.cpu().numpy() if orig_numpy else x_t

    def __str__(self):
        return f"ObsPyCosineTaper(max_percentage={self.max_percentage}, max_length={self.max_length})"