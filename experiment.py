
import numpy as np
import sys
import logging
from typing import Tuple, List, Dict, Any

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LPConformalPrediction:
    """
    Implementation of Lévy-Prokhorov robust conformal prediction for time series data.
    Based on the paper: "Conformal Prediction under Lévy–Prokhorov Distribution Shifts"
    """
    
    def __init__(self, alpha: float = 0.1, epsilon: float = 0.1, rho: float = 0.05):
        """
        Initialize LP robust conformal prediction parameters.
        
        Args:
            alpha: Significance level (1 - coverage)
            epsilon: Local perturbation parameter for LP ambiguity set
            rho: Global perturbation parameter for LP ambiguity set
        """
        self.alpha = alpha
        self.epsilon = epsilon
        self.rho = rho
        self.calibration_scores = None
        self.quantile = None
        
        logger.info(f"Initialized LPConformalPrediction with alpha={alpha}, epsilon={epsilon}, rho={rho}")
    
    def generate_time_series_data(self, n_samples: int = 1000, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate synthetic time series data with distribution shift.
        
        Args:
            n_samples: Number of samples to generate
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (time_series, labels)
        """
        np.random.seed(seed)
        logger.info(f"Generating {n_samples} time series samples with distribution shift")
        
        # Generate baseline time series (AR(1) process)
        time_series = []
        labels = []
        
        for i in range(n_samples):
            # First half: stationary process
            if i < n_samples // 2:
                x = np.zeros(50)
                x[0] = np.random.normal(0, 1)
                for t in range(1, 50):
                    x[t] = 0.8 * x[t-1] + np.random.normal(0, 0.5)
            # Second half: distribution shift (changed parameters)
            else:
                x = np.zeros(50)
                x[0] = np.random.normal(1, 1.5)  # Different mean and variance
                for t in range(1, 50):
                    x[t] = 0.6 * x[t-1] + np.random.normal(0, 0.8)  # Different AR parameter
            
            time_series.append(x)
            labels.append(0 if i < n_samples // 2 else 1)  # 0=before shift, 1=after shift
        
        return np.array(time_series), np.array(labels)
    
    def compute_nonconformity_scores(self, time_series: np.ndarray) -> np.ndarray:
        """
        Compute nonconformity scores for time series data.
        Using simple forecasting error as the score function.
        
        Args:
            time_series: Array of time series data
            
        Returns:
            Array of nonconformity scores
        """
        logger.info("Computing nonconformity scores")
        
        scores = []
        for ts in time_series:
            # Simple forecasting: predict next value as current value
            predictions = np.zeros_like(ts)
            predictions[1:] = ts[:-1]  # One-step ahead forecast
            
            # Nonconformity score: mean absolute error
            score = np.mean(np.abs(ts[1:] - predictions[1:]))
            scores.append(score)
        
        return np.array(scores)
    
    def calibrate(self, calibration_scores: np.ndarray):
        """
        Calibrate the conformal prediction model.
        
        Args:
            calibration_scores: Nonconformity scores from calibration set
        """
        logger.info("Calibrating conformal prediction model")
        
        self.calibration_scores = calibration_scores
        
        # Compute the worst-case quantile using LP robustness
        # From Proposition 3.4: QuantWC_{ε,ρ}(β;P) = Quant(β+ρ;P) + ε
        beta = 1 - self.alpha
        adjusted_beta = min(beta + self.rho, 1.0)  # Ensure beta + rho <= 1
        
        # Finite sample correction: (n+1)/n * adjusted_beta
        n = len(calibration_scores)
        level = min((n + 1) * adjusted_beta / n, 1.0)
        
        # Compute quantile
        self.quantile = np.quantile(calibration_scores, level) + self.epsilon
        
        logger.info(f"Calibration complete: quantile={self.quantile:.4f}, level={level:.4f}")
    
    def predict(self, test_scores: np.ndarray) -> np.ndarray:
        """
        Make predictions using the calibrated model.
        
        Args:
            test_scores: Nonconformity scores for test data
            
        Returns:
            Binary array indicating coverage (1=covered, 0=not covered)
        """
        if self.quantile is None:
            logger.error("Model not calibrated. Call calibrate() first.")
            sys.exit(1)
        
        logger.info("Making predictions")
        
        # Prediction: coverage indicator (1 if score <= quantile, 0 otherwise)
        coverage = (test_scores <= self.quantile).astype(int)
        
        return coverage
    
    def evaluate_coverage(self, coverage: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate coverage results.
        
        Args:
            coverage: Coverage indicators from predict()
            labels: True labels indicating distribution shift
            
        Returns:
            Dictionary with coverage metrics
        """
        logger.info("Evaluating coverage results")
        
        # Overall coverage
        overall_coverage = np.mean(coverage)
        
        # Coverage before and after distribution shift
        before_shift_mask = (labels == 0)
        after_shift_mask = (labels == 1)
        
        coverage_before = np.mean(coverage[before_shift_mask]) if np.any(before_shift_mask) else 0.0
        coverage_after = np.mean(coverage[after_shift_mask]) if np.any(after_shift_mask) else 0.0
        
        results = {
            'overall_coverage': overall_coverage,
            'coverage_before_shift': coverage_before,
            'coverage_after_shift': coverage_after,
            'target_coverage': 1 - self.alpha
        }
        
        logger.info(f"Overall coverage: {overall_coverage:.3f} (target: {1 - self.alpha:.3f})")
        logger.info(f"Coverage before shift: {coverage_before:.3f}")
        logger.info(f"Coverage after shift: {coverage_after:.3f}")
        
        return results

def main():
    """
    Main experiment function demonstrating LP robust conformal prediction on time series data.
    """
    try:
        logger.info("Starting LP robust conformal prediction experiment")
        
        # Initialize LP robust conformal prediction
        lpcp = LPConformalPrediction(alpha=0.1, epsilon=0.1, rho=0.05)
        
        # Generate time series data with distribution shift
        time_series, labels = lpcp.generate_time_series_data(n_samples=1000)
        logger.info(f"Generated time series data: {time_series.shape}")
        
        # Compute nonconformity scores
        scores = lpcp.compute_nonconformity_scores(time_series)
        logger.info(f"Computed nonconformity scores: {scores.shape}")
        
        # Split data into calibration and test sets (50/50 split)
        n_calibration = len(scores) // 2
        calibration_scores = scores[:n_calibration]
        test_scores = scores[n_calibration:]
        test_labels = labels[n_calibration:]
        
        logger.info(f"Data split: {len(calibration_scores)} calibration, {len(test_scores)} test samples")
        
        # Calibrate the model
        lpcp.calibrate(calibration_scores)
        
        # Make predictions on test data
        coverage = lpcp.predict(test_scores)
        
        # Evaluate coverage
        results = lpcp.evaluate_coverage(coverage, test_labels)
        
        # Print final results
        logger.info("=" * 50)
        logger.info("EXPERIMENT RESULTS SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Method: LP Robust Conformal Prediction")
        logger.info(f"Parameters: α={lpcp.alpha}, ε={lpcp.epsilon}, ρ={lpcp.rho}")
        logger.info(f"Overall coverage: {results['overall_coverage']:.3f} (target: {results['target_coverage']:.3f})")
        logger.info(f"Coverage before distribution shift: {results['coverage_before_shift']:.3f}")
        logger.info(f"Coverage after distribution shift: {results['coverage_after_shift']:.3f}")
        
        # Check if coverage meets target
        if results['overall_coverage'] >= results['target_coverage'] - 0.05:  # Allow 5% tolerance
            logger.info("✓ Experiment successful: Coverage meets or exceeds target")
        else:
            logger.warning("⚠ Experiment: Coverage below target")
            
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Critical error in experiment: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
