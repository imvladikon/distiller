class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def dict_from_var_names(*args):
     return dict(((k, eval(k)) for k in args))

