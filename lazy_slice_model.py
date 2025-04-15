import os
import torch
import logging
import pickle
import psutil
import shutil
from collections import OrderedDict
from typing import Type


class LazySliceModel:
    """
    a model manager that loads and unloads slices (model parts) on-demand.
    supports multi-level caching: GPU -> CPU -> Disk.
    """

    def __init__(
        self,
        slice_constructors: dict[int, tuple[type[torch.nn.Module], dict]],
        cache_dir: str = "slyce_cache",
        gpu_access: bool = True,
        gpu_limit_mb: float | None = None,
        cpu_limit_mb: float | None = None
    ) -> None:
        """
        description:
            constructor for LazySliceModel.
        args:
            slice_constructors: mapping of slice_id to (class, kwargs).
            cache_dir: directory for saving/loading slices to/from disk. should be a relative path. 
            gpu_access: is it allowed to use the gpu?
            gpu_limit_mb: limit on gpu usage in MB.
            cpu_limit_mb: limit on cpu usage in MB.
        note:
            these limits are on the model parameters, and NOT on the overall 
            memory usage.
            plan is to provide some api, to check and decide on these limits
            later. and then assume that the limits would be followed
        """

        # simple assert
        assert (not gpu_access) or (gpu_access and torch.cuda.is_available()), "gpu_access is set to true, but torch.cuda.is_available() is false."
        
        # save attributes
        self.slice_constructors = slice_constructors
        self.cache_dir = cache_dir
        self.gpu_access = gpu_access
        self.gpu_limit_mb = gpu_limit_mb
        self.cpu_limit_mb = cpu_limit_mb
    
        # usage tracking
        self.gpu_memory_usage = 0 
        self.cpu_memory_usage = 0
    
        # slice handling attributes
        self.active_slice = None
        self.active_slice_id = None
        self.cpu_slices = OrderedDict()
    
        # logger
        self.logger = None

    

    def set_logger(self, logger: logging.Logger) -> None:
        """
        description:
            method to set logger.
        args:
            logger: a logger for logging messages.
        """
        self.logger = logger

    # UNFINISHED
    def set_slice_id(self, slice_id: int) -> None:
        """
        description:
            method to set/activate a slice using slice id.
        args:
            slice_id: id of the slice to activate.
        """
        # simple check
        if self.active_slice_id == slice_id:
            return

        # case 1: active slice id is None, basically nothing has been started
        if self.active_slice_id is None:

            try:
                cls, kwargs = self.slice_constructors[slice_id]
                slice_model = cls(**kwargs).to('cpu')
                self.logger.info(f"set_slice_id: instantiated slice {slice_id} in CPU.")

            except Exception as e:
                self.logger.info(f"set_slice_id: error on instantiating slice {slice_id} in CPU.")
                self.logger.info(f"error msg: {e}")

            # case 1a: gpu_access is false
            if self.gpu_access == False:

                # update active slice and active slice id
                self.active_slice = slice_model
                self.active_slice_id = slice_id

                # update memory information
                self.cpu_memory_usage += self._estimate_tensor_memory(self.active_slice)

                # enforce the cpu memory limits
                if self.cpu_memory_usage > self.cpu_limit_mb:
                    # offload slice
                    self.offload_slice(self.active_slice_id)

                    # log  
                    self.logger.info("set_slice_id: CPU memory limit exceeded, offloading the slice {self.active_slice_id}.")
                    
                    # update active slice and active slice id
                    self.active_slice = None 
                    self.active_slice_id = None
                else:
                    self.logger.info(f"set_slice_id: slice {self.active_slice_id} initialization complete [CPU].")

            # case 1b: gpu_access is true
            else:
                try:
                    slice_model.to("cuda")
                    
                    self.active_slice = slice_model
                    self.active_slice_id = slice_id

                    # update memory information
                    self.gpu_memory_usage += self._estimate_tensor_memory(self.active_slice)

                    # enforce the gpu memory limits
                    if self.gpu_memory_usage > self.gpu_limit_mb:
                        # offload slice, this will go to cpu
                        self.offload_slice(self.active_slice_id)
                        # log  
                        self.logger.info("set_slice_id: GPU memory limit exceeded, offloading the slice {self.active_slice_id} to CPU.")

                        # TODO enforce cpu limit
                    else:
                        self.logger.info(f"set_slice_id: slice {self.active_slice_id} initialization complete [GPU].")

                except Exception as e:
                    self.logger.info("set_slice_id: error while sending slice to GPU.")
                    self.logger.info("error msg: {e}")

        # case 2: active slice id is not none, there was someone before this
        else:
            # case 2a: gpu_access is true
            if self.gpu_access == True:
                # TODO: 
                _ = _
                self.offload_slice(self.active_slice_id)
                self._enforce_cpu_limit()

                
                # self.active_slice_id = None 
                # self.active_slice = None 


            # case 2b: gpu_access is false
            else:
                self.offload_slice(self.active_slice_id)
                self.set_slice_id(slice_id)

    def _estimate_tensor_memory(self, model: torch.nn.Module) -> float:
        """
        description:
            method to estimat the memory usage of a model in MB.
        args:
            model: model whose memory usage needs to be found.
        """
        return sum(p.nelement() * p.element_size() for p in model.parameters()) / (1024 * 1024)
                
    def offload_slice(self, slice_id: int) -> None:
        """
        description:
            method to move a slice, GPU to CPU, CPU to Disk.
        args:
            slice_id: id of the slice to offload.
        note:
            also handles the cpu_memory_usage and gpu_memory_usage variables
        """
        _ = _



        
        

    