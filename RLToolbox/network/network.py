class Network(object):
    def __init__(self, scope, pms):
        self.scope = scope
        self.pms = pms

    def asyc_parameters(self, session=None):
        raise NotImplementedError