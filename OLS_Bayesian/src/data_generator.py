import numpy as np

class DataGenerator():
    def __init__(self, random_state=None):
        # Use proper random state handling with Generator
        self.rng = np.random.Generator(np.random.PCG64(random_state))
    
    def get_linear_regression_data(self, n_samples, n_features):
        """Generate linear regression data with normal distribution."""
        X = self.rng.standard_normal((n_samples, n_features))
        true_theta = self.rng.standard_normal(n_features)
        y = X @ true_theta
        return X, y, true_theta

    def get_fft_regression_data(self, n_samples, n_features):
        """Generate regression data using FFT-transformed features."""
        X = self.rng.standard_normal((n_samples, n_features))
        # Apply FFT to create frequency-domain features
        X_fft = np.fft.fft(X, axis=1).real
        true_theta = self.rng.standard_normal(n_features)
        y = X_fft @ true_theta
        return X_fft, y, true_theta
    
    def get_square_wave_data(self, n_samples, n_features):
        """Generate square wave data."""
        #for a better visualization, we will not randomly generate X
        harmonic_freqs = np.linspace(1, 10, n_features)
        # Generate a time vector
        t = np.linspace(0, 1, n_samples)
        # Create a square wave signal using the harmonic frequencies
        square_wave = np.sign(np.sin(2 * np.pi * harmonic_freqs[:, None] * t[None, :]))
        # Reshape to match the number of features
        square_wave = square_wave.T  # Transpose to have shape (n_samples, n_features)
        # Generate the input features
        n_features = square_wave.shape[1]
        # Generate random noise
        # noise = self.rng.standard_normal((n_samples, n_features)) * 0.1  # Adjust noise level as needed
        # Combine square wave with noise
        X = square_wave #+ noise
        # Create target variable as a linear combination of features
        # Here we can use a simple linear combination with random coefficients
        #geneate target variable y, using square wave as features
        true_theta = np.ones(n_features)
        y = X @ true_theta
        #y = np.sign (np.sin(2 * np.pi * harmonic_freqs[:, None] * t[None, :])).T 

        #solve the regression problem to get the true theta
        #true_theta = np.linalg.lstsq(X, y, rcond=None)[0]
        # Return the generated data
        return X, y, true_theta
       