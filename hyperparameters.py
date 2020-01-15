class Hyperparameters:
    def __init__(self):
        self.hyperparameters={}

    def add(self,name,value):
        self.hyperparameters[name]=value

    def add_list(self,name_value_list):
        for [name,value] in name_value_list:
            self.hyperparameters[name]=value

    def get(self,name):
        if name in self.hyperparameters:
            return self.hyperparameters[name]
        return None
    
    def get_list(self):
        return self.hyperparameters


