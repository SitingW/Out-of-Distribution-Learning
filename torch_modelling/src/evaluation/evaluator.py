class Evaluator:
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
    
    def evaluate(self, test_loader, criterion):
        """Evaluate model on test set."""
        self.model.eval()
        predictions = []
        targets = []
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                loss = criterion(outputs, target)
                total_loss += loss.item()
                
                predictions.append(outputs.cpu().numpy())
                targets.append(target.cpu().numpy())
        
        predictions = np.concatenate(predictions, axis=0)
        targets = np.concatenate(targets, axis=0)
        
        # Calculate metrics
        metrics = calculate_metrics(targets, predictions)
        metrics['test_loss'] = total_loss / len(test_loader)
        
        return metrics, predictions, targets   