from .regression_metrics import mse, rmse, mae, r2, mape, huber_loss
#from .classification_metrics import accuracy, precision, recall, f1, auc_roc
#from .custom_metrics import custom_weighted_mse, percentile_loss, directional_accuracy

class MetricRegistry:
    """Registry for all available metrics"""
    
    def __init__(self):
        self.regression_metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'mape': mape,
            'huber_loss': huber_loss
        }
        
        # self.classification_metrics = {
        #     'accuracy': accuracy,
        #     'precision': precision,
        #     'recall': recall,
        #     'f1': f1,
        #     'auc_roc': auc_roc
        # }
        
        # self.custom_metrics = {
        #     'custom_weighted_mse': custom_weighted_mse,
        #     'percentile_loss': percentile_loss,
        #     'directional_accuracy': directional_accuracy
        # }
        
        # Combined registry
        self.all_metrics = {
            **self.regression_metrics,
        #    **self.classification_metrics,
        #    **self.custom_metrics
        }
    
    def get_metric(self, name):
        """Get a specific metric function by name"""
        if name not in self.all_metrics:
            raise ValueError(f"Metric '{name}' not found. Available metrics: {list(self.all_metrics.keys())}")
        return self.all_metrics[name]
    
    def calculate_metrics(self, y_true, y_pred, metric_names):
        """Calculate multiple metrics"""
        results = {}
        for name in metric_names:
            metric_func = self.get_metric(name)
            results[name] = metric_func(y_true, y_pred)
        return results
    
    def register_custom_metric(self, name, metric_func):
        """Register a new custom metric"""
        self.all_metrics[name] = metric_func
        self.custom_metrics[name] = metric_func
    
    def list_available_metrics(self):
        """List all available metrics by category"""
        return {
            'regression': list(self.regression_metrics.keys()),
            # 'classification': list(self.classification_metrics.keys()),
            # 'custom': list(self.custom_metrics.keys())
        }