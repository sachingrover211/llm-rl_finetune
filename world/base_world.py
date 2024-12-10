class BaseWorld:
    def __init__(self, name):
        self.name = name
    
    def run(self):
        raise NotImplementedError("run method not implemented")
    
    

