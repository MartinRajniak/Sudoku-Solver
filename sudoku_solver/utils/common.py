def singleton(cls):
    """Decorator that makes sure there is only one instance of a class"""
    _instances = {}

    def getinstance(*args, **kwargs):
        if cls not in _instances:
            _instances[cls] = cls(*args, **kwargs)
        return _instances[cls]

    return getinstance
