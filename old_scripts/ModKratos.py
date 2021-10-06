import os
import sys
import inspect

import KratosMultiphysics as KMP
import KratosMultiphysics.FluidDynamicsApplication as FDA

def print_imported_modules():
    loaded_modules = [mod for mod in sys.modules.keys() if '__file__' in dir(sys.modules[mod]) and sys.modules[mod].__file__]
    kratos_modules = [mod for mod in loaded_modules if "KratosMultiphysics" in sys.modules[mod].__file__]

    kratos_apps = [mod for mod in kratos_modules if os.path.basename(sys.modules[mod].__file__) == "__init__.py"]

    return kratos_apps

loaded_modules = print_imported_modules()

for mod in loaded_modules:
    print("Loaded module:", mod)
