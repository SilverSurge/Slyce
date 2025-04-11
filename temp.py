# imports
import os
import torch
import psutil
import logging
import shutil
from typing import Dict, Tuple, Optional, Callable, Any
from collections import OrderedDict

class LazySliceModel:
    """
    a model manager that loads and unloads slices (model parts) on-demand.
    supports multi-level caching: GPU -> CPU -> Disk.
    """

    def __init__(
        self,
        slice_constructors: Dict[str, Tuple[Type[torch.nn.Module], dict]],
        cache_dir: str = "slyce_cache",
        gpu_strategy: str = "active_only",
        gpu_limit_mb: Optional[float] = None,
        cpu_limit_mb: Optional[float] = None
    ) -> None:
        """
        Args:
            slice_constructors (dict): mapping of slice_id to (class, kwargs).
            cache_dir (str): directory for saving/loading slices to/from disk.
            gpu_strategy (str): 'none', 'active_only', or 'limit_gpu'.
            gpu_limit_mb (float): max GPU memory in MB before evicting to CPU.
            cpu_limit_mb (float): max CPU memory in MB before saving to disk.
        """

        # save and initialize attributes
        self.slice_constructors = slice_constructors
        self.cache_dir = cache_dir
        self.slices = {}    # All initialized slices
        self.active_id = None

        # strategy and limits
        self.gpu_strategy = gpu_strategy
        self.gpu_limit_mb = gpu_limit_mb
        self.cpu_limit_mb = cpu_limit_mb

        # caches and usage tracking
        self.gpu_cache = OrderedDict()
        self.cpu_cache = OrderedDict()
        self.gpu_memory_usage = 0.0
        self.cpu_memory_usage = 0.0

        os.makedirs(self.cache_dir, exist_ok=True)
        self._setup_logger()

    def _setup_logger(self) -> None:
        """initializes logging to file and console."""
        self.logger = logging.getLogger("Slyce")
        self.logger.setLevel(logging.INFO)


        # avoid duplicate handlers
        if not self.logger.handlers:
            console_handler = logging.StreamHandler()
            file_handler = logging.FileHandler(os.path.join(self.cache_dir, "slyce.log"))

            formatter = logging.Formatter("[%(levelname)s] %(message)s")
            console_handler.setFormatter(formatter)
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))

            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)
    
    def _estimate_tensor_memory(self, model: torch.nn.Module) -> float:
        """estimates the memory usage of a model in MB."""
        return sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)
    
    def _save_to_disk(self, slice_id: int, model: torch.nn.Module) -> None:
        """saves the model's state_dict to disk."""
        path = os.path.join(self.cache_dir, f"{slice_id}.pt")
        torch.save(model.state_dict(), path)

    def _load_from_disk(self, slice_id: int) -> bool:
        """loads state_dicts into slices from a saved file."""
        try:
            saved = torch.load(path, map_location="cpu")
            for k, state in saved.items():
                if k not in self.slices:
                    cls, kwargs = self.slice_constructors[k]
                    self.slices[k] = cls(**kwargs)
                self.slices[k].load_state_dict(state)
            self.logger.info(f"Loaded state_dict from {path}")
        except Exception as e:
            self.logger.error(f"Failed to load state_dict from {path}: {e}")

model = LazySliceModel()
""" 

class LazySliceModel:
    def __init__(
        self,
        slice_constructors: dict[int, tuple[type[torch.nn.Module], dict]],
        cache_dir: str = "slyce_cache",
        gpu_strategy: str = "active_only",
        gpu_limit_mb: float | None = None,
        cpu_limit_mb: float | None = None
    ) -> None:
        ...

    def set_slice_id(self, slice_id: int) -> None:
        ...

    def forward(self, **kwargs) -> torch.Tensor:
        ...

    def save_state_dict(self, path: str) -> None:
        ...

    def load_state_dict(self, path: str) -> None:
        ...

    def _save_to_disk(self, slice_id: int) -> None:
        ...

    def _load_from_disk(self, slice_id: int) -> torch.nn.Module:
        ...

    def erase_slice(self, slice_id: int) -> None:
        ...

    def log_cpu_usage(self) -> None:
        ...

    def log_gpu_usage(self) -> None:
        ...

    def summary(self) -> None:
        ...

    def _enforce_gpu_limit(self) -> None:
        ...

    def _enforce_cpu_limit(self) -> None:
        ...

    def _estimate_tensor_memory(self, model: torch.nn.Module) -> float:
        ...

"""


""" 
 def __init__(
        self,
        slice_constructors: dict[int, tuple[type[torch.nn.Module], dict]],
        cache_dir: str = "slyce_cache",
        gpu_access: bool = True,
        gpu_limit_mb: float | None = None,
        cpu_limit_mb: float | None = None
    ) -> None:



def set_logger(self, logger: logging.Logger) -> None:

def set_slice_id(self, slice_id: int) -> None:

def forward(self, **kwargs) -> torch.Tensor:
    **kwargs handles the different forward signatures of different slices, it may so happen that one slice needs 
    forward(self, x) and other needs forward(self, x, y)

def reset_slice(self, slice_id: int) -> None:
    move slice if in gpu to cpu, if in cpu to disk, if in disk then delete
    maybe get a better name for this function, demote slice (but this doesn't sound nice)

def save_state_dict(self, path) -> None:
    save the state of the current slice to the path

def load_state_dict(self, path) -> None:
    load the state for the path to the slice, 

def _verify_gpu_limit(self) -> bool:
    assuming gpu is available:
    when we set slice_id, first the model would be loaded in cpu, so the active_slice is made
    (it is an assumption that you know that your slice atleasts fits in the cpu memory)
    check if the active slice fits in the gpu limit, if it does then move the 
    previous active slice to the cpu, 
    (and then enforce cpu limit, which would probably be implemented in set slice id)

def _enforce_cpu_limit(self) -> None:
    assuming gpu is available:
    now we have some slices in the cpu memory, but we also have a limit,
    so till we are above the limit, reset_slice for the least recently used slices
    (basically, send them to disk).

    (note this will always be called in set_slice_id)

    if gpu is not available:
    then we only have one slice in the cpu, as if we did have two or more, then 
    it would be better to create larger slices. which is the premise.

def _estimate_tensor_memory(self, model: torch.nn.Module) -> float:
    
def log_cpu_usage(self) -> None:

def log_gpu_usage(self) -> None:

def summary(self) -> None:


Note:
let say I have 6GB of cpu memory, and my largest Slice takes about 800MB of space
then for I should set a cpu limit to 6GB - 800MB, or maybe just to be safe 6GB - (1.5)*800MB
so that there will be no memory errors when enforcing cpu limits

_enforce_cpu_limits and _verify_gpu_limits are only useful in case of gpu access.
    
"""