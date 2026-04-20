from __future__ import annotations
from open3d.cpu import pybind
from open3d.cpu.pybind import camera
from open3d.cpu.pybind import core
from open3d.cpu.pybind import data
from open3d.cpu.pybind import geometry
from open3d.cpu.pybind import io
from open3d.cpu.pybind import pipelines
from open3d.cpu.pybind import t
from open3d.cpu.pybind import utility
import os
import platform as platform
import re as re
from . import cpu
from . import ml
from . import visualization
__all__: list[str] = ['camera', 'core', 'cpu', 'data', 'geometry', 'io', 'ml', 'open3d', 'pipelines', 'platform', 'pybind', 're', 't', 'utility', 'visualization']
def _jupyter_labextension_paths():
    """
    Called by Jupyter Lab Server to detect if it is a valid labextension and
        to install the widget.
    
        Returns:
            src: Source directory name to copy files from. Webpack outputs generated
                files into this directory and Jupyter Lab copies from this directory
                during widget installation.
            dest: Destination directory name to install widget files to. Jupyter Lab
                copies from `src` directory into <jupyter path>/labextensions/<dest>
                directory during widget installation.
        
    """
def _jupyter_nbextension_paths():
    """
    Called by Jupyter Notebook Server to detect if it is a valid nbextension
        and to install the widget.
    
        Returns:
            section: The section of the Jupyter Notebook Server to change.
                Must be "notebook" for widget extensions.
            src: Source directory name to copy files from. Webpack outputs generated
                files into this directory and Jupyter Notebook copies from this
                directory during widget installation.
            dest: Destination directory name to install widget files to. Jupyter
                Notebook copies from `src` directory into
                <jupyter path>/nbextensions/<dest> directory during widget
                installation.
            require: Path to importable AMD Javascript module inside the
                <jupyter path>/nbextensions/<dest> directory.
        
    """
__DEVICE_API__: str = 'cpu'
__version__: str = '0.19.0'
_build_config: dict = {'BUILD_TENSORFLOW_OPS': False, 'BUILD_PYTORCH_OPS': False, 'BUILD_CUDA_MODULE': False, 'BUILD_SYCL_MODULE': False, 'BUILD_AZURE_KINECT': True, 'BUILD_LIBREALSENSE': True, 'BUILD_SHARED_LIBS': False, 'BUILD_GUI': True, 'ENABLE_HEADLESS_RENDERING': False, 'BUILD_JUPYTER_EXTENSION': True, 'BUNDLE_OPEN3D_ML': False, 'GLIBCXX_USE_CXX11_ABI': True, 'CMAKE_BUILD_TYPE': 'Release', 'CUDA_VERSION': '', 'CUDA_GENCODES': '', 'Tensorflow_VERSION': '', 'Pytorch_VERSION': '', 'WITH_OPENMP': True}
_win32_dll_dir: os._AddedDllDirectory  # value = <AddedDllDirectory()>
open3d = 
