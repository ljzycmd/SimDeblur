# Registry Class
# CMD 
# Refer this in Detectron2

class Registry:
    def __init__(self, name):
        self._name = name
        self._obj_map = {}
    
    def _do_register(self, name, obj):
        assert (name not in self._obj_map), "The object named: {} was already registered in {} registry! ".format(name, self._name)
        self._obj_map[name] = obj

    def register(self, obj=None):
        """
        Register the given object under the name obj.__name__.
        Can be used as either a decorator or not.
        """
        if obj is None:
            # used as a decorator
            def deco(func_or_class):
                name = func_or_class.__name__
                self._do_register(name, func_or_class)
                return func_or_class
            
            return deco
        
        name = obj.__name__
        self._do_register(name, obj)
    
    def get(self, name):
        ret = self._obj_map.get(name)
        if ret is None:
            raise KeyError("No object names {} found in {} registry!".format(name, self._name))

        return ret
    
    def __getitem__(self, name):
        return self.get(name)
    
    def keys(self):
        return self._obj_map.keys()