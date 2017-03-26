from darkopt.optimize.engine.random_search import RandomSearch  # NOQA


def get_engine(engine, param_space):
    if not isinstance(engine, str):
        return engine

    if engine == 'random_search':
        return RandomSearch(param_space)

    raise ValueError('Unknown optimizer engine: {}'.format(engine))
