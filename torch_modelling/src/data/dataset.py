
class SupervisedDataset(Dataset):
    """Generic dataset for supervised learning (regression or classification)."""
    
    def __init__(self, X, y, transform=None, target_transform=None):
        self.X = torch.FloatTensor(X)
        
        # Handle different target types
        if isinstance(y, (list, np.ndarray)):
            # For regression: keep as float
            # For classification: convert to long if integers
            if np.issubdtype(np.array(y).dtype, np.integer):
                self.y = torch.LongTensor(y)  # Classification labels
            else:
                self.y = torch.FloatTensor(y)  # Regression targets
        else:
            self.y = y
        
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        sample = self.X[idx]
        target = self.y[idx]
        
        if self.transform:
            sample = self.transform(sample)
        
        if self.target_transform:
            target = self.target_transform(target)
        
        return sample, target

class RegressionDataset(SupervisedDataset):
    """Dataset specifically for regression tasks."""
    
    def __init__(self, X, y, transform=None, target_transform=None):
        # Ensure targets are float for regression
        y = np.array(y, dtype=np.float32)
        super().__init__(X, y, transform, target_transform)

class ClassificationDataset(SupervisedDataset):
    """Dataset specifically for classification tasks."""
    
    def __init__(self, X, y, transform=None, target_transform=None, num_classes=None):
        # Ensure targets are integers for classification
        y = np.array(y, dtype=np.int64)
        super().__init__(X, y, transform, target_transform)
        self.num_classes = num_classes or len(np.unique(y))
    
    def get_class_weights(self):
        """Calculate class weights for imbalanced datasets."""
        unique, counts = np.unique(self.y.numpy(), return_counts=True)
        total_samples = len(self.y)
        weights = total_samples / (len(unique) * counts)
        return torch.FloatTensor(weights)

class TimeSeriesDataset(Dataset):
    """Dataset for time series data with sequence windows."""
    
    def __init__(self, data, sequence_length, prediction_length=1, transform=None):
        self.data = torch.FloatTensor(data)
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.transform = transform
    
    def __len__(self):
        return len(self.data) - self.sequence_length - self.prediction_length + 1
    
    def __getitem__(self, idx):
        sequence = self.data[idx:idx + self.sequence_length]
        target = self.data[idx + self.sequence_length:idx + self.sequence_length + self.prediction_length]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence, target

def create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, 
                       batch_size=32, task_type='regression'):
    """Create data loaders for train/val/test sets."""
    
    # Choose appropriate dataset class based on task type
    if task_type == 'regression':
        DatasetClass = RegressionDataset
    elif task_type == 'classification':
        DatasetClass = ClassificationDataset
    else:
        DatasetClass = SupervisedDataset
    
    train_dataset = DatasetClass(X_train, y_train)
    val_dataset = DatasetClass(X_val, y_val)
    test_dataset = DatasetClass(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def create_regression_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    """Convenience function for regression tasks."""
    return create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, 
                              batch_size, task_type='regression')

def create_classification_loaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    """Convenience function for classification tasks."""
    return create_data_loaders(X_train, y_train, X_val, y_val, X_test, y_test, 
                              batch_size, task_type='classification')
