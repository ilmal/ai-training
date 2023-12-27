import os
import pandas as pd
import importlib

def get_models_list():
    files = os.listdir("/models")

    models = []
    for file in files:
        if not "model_" in file or not ".py" in file or "template" in file: continue 
        
        model = file.split(".")[0].split("_")[1]
        models.append(model)
    
    return models

def check_completed_models(models):
    return pd.read_csv("/results/model_data.csv")["model_generation"].tolist()

def import_function(module_name, function_name):
    try:
        # Build the full path to the module
        module_path = f"/models/{module_name}"
        
        # Check if the module file exists
        if not os.path.isfile(module_path):
            raise ImportError(f"Module file {module_path} not found.")

        # Load the module spec
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        
        # Import the module dynamically
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the function from the module
        func = getattr(module, function_name)

        return func
    except ImportError as e:
        print(f"Error: {e}")
    except AttributeError:
        print(f"Error: Function {function_name} not found in module {module_name}.")


def get_model():
    models = [int(e) for e in get_models_list()]
    completed_models = [int(e) for e in check_completed_models(models)]

    # print(models)
    # print(completed_models)

    if len(models) == len(completed_models): raise Exception("NO MODEL TO TRAIN")
    
    difference = list(set(models) - set(completed_models))

    print(difference)

    difference_str =  []
    for e in difference:
        if e < 10:
            e = "0"+str(e)
        else:
            e = str(e)
        difference_str.append(e)

    if not len(difference) > 0: raise Exception("NO MODEL TO TRAIN")

    model_function = import_function(f"model_{difference_str[0]}.py", "model")

    return difference_str[0], model_function()

    


