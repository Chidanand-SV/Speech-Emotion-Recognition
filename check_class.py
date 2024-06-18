import pickle

# Function to load the pickle file and check classification names
def load_and_check_classification_names(pkl_file_path):
    # Load the pickle file
    with open(pkl_file_path, 'rb') as file:
        model = pickle.load(file)
    
    # Check if the loaded object has 'classes_' attribute
    if hasattr(model, 'classes_'):
        classification_names = model.classes_
        print("Classification names in the model:", classification_names)
    else:
        print("The loaded object does not have 'classes_' attribute.")

# Example usage
pkl_file_path = 'new.pkl'  # Replace with your actual file path
load_and_check_classification_names(pkl_file_path)
