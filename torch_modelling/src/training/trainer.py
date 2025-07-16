
class Trainer:
    def __init__(self, model, optimizer, criterion, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(device)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            data, target = data.to(self.device), target.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(data)
            loss = self.criterion(outputs, target)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def train(self, train_loader, num_epochs, val_loader=None):
        """Full training loop."""
        train_losses = []
        val_losses = []
        
        for epoch in range(num_epochs):
            # Training
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Validation
            if val_loader:
                val_loss = self.validate(val_loader)
                val_losses.append(val_loss)
                self.logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            else:
                self.logger.info(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}')
        
        return train_losses, val_losses
    
    def validate(self, val_loader):
        """Validation step."""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.model(data)
                loss = self.criterion(outputs, target)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
