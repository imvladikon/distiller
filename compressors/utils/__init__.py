from typing import Union, Dict

from .data import any2device


def set_requires_grad(model, requires_grad: Union[bool, Dict[str, bool]]):
    """Sets the ``requires_grad`` value for all model parameters.

    Example::

        >>> model = SimpleModel()
        >>> set_requires_grad(model, requires_grad=True)
        >>> # or
        >>> model = SimpleModel()
        >>> set_requires_grad(model, requires_grad={""})

    Args:
        model: model
        requires_grad: value
    """
    if isinstance(requires_grad, dict):
        for name, param in model.named_parameters():
            assert name in requires_grad, f"Parameter `{name}` does not exist in requires_grad"
            param.requires_grad = requires_grad[name]
    else:
        requires_grad = bool(requires_grad)
        for param in model.parameters():
            param.requires_grad = requires_grad

__all__ = ["any2device", "set_requires_grad"]
