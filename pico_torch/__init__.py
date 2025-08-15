from .tensor import Tensor

# Make Tensor available directly in the package namespace
__all__ = ["Tensor"]

# Add Tensor to builtins so it's available everywhere
import builtins
builtins.Tensor = Tensor
