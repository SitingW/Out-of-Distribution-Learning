from .data_generation import DataGenerator
from .data_loading import DataLoader
from .data_splitting import DataSplitter
from .preprocessing import DataPreprocessor

def create_complete_dataset_pipeline(data_source: str = 'synthetic',
                                   task_type: str = 'regression',
                                   **kwargs):
    """Complete pipeline from data generation/loading to dataset creation"""
    
    # Step 1: Generate data
    if data_source == 'synthetic':
        if task_type == 'regression':
            X, y = DataGenerator.generate_linear_regression(**kwargs)
        else:
            X, y = DataGenerator.generate_classification_data(**kwargs)
    # elif data_source == 'file':
    #     X, y = DataLoader.load_csv(kwargs['file_path'], kwargs['target_column'])
    # else:
    #     raise ValueError(f"Unknown data source: {data_source}")

    """
    I don't need to split data nor preprocess the data, as they are all generated.
    but do I need a dataloader? what is the purpose of dataloader here? somewhat unnecessary?
    """