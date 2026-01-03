"""
Cut-based baseline trigger model.

Traditional trigger approach using simple selection criteria.
"""

import numpy as np
import pandas as pd
from typing import Dict
import time


class CutBasedTrigger:
    """Traditional cut-based trigger implementation."""
    
    def __init__(self, config: Dict):
        """
        Initialize cut-based trigger.
        
        Args:
            config: Configuration dictionary with baseline parameters
        """
        self.config = config
        self.pt_threshold = config['pt_threshold']
        self.eta_max = config['eta_max']
        self.chi2_max = config['chi2_max']
        self.min_hits = config['min_hits']
        self.isolation_max = config['isolation_max']
        
        self.is_trained = True  # No training needed for cuts
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Apply trigger cuts to events.
        
        Args:
            X: DataFrame with event features
            
        Returns:
            Binary predictions (1 = accept, 0 = reject)
        """
        # Apply all cuts (AND logic)
        mask = (
            (X['pt'] >= self.pt_threshold) &
            (np.abs(X['eta']) <= self.eta_max) &
            (X['chi2'] <= self.chi2_max) &
            (X['n_hits'] >= self.min_hits) &
            (X['isolation'] <= self.isolation_max)
        )
        
        return mask.astype(int).values
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Return probabilities (hard 0 or 1 for cut-based).
        
        Args:
            X: DataFrame with event features
            
        Returns:
            Array of shape (n_samples, 2) with class probabilities
        """
        predictions = self.predict(X)
        proba = np.zeros((len(predictions), 2))
        proba[:, 1] = predictions
        proba[:, 0] = 1 - predictions
        
        return proba
    
    def get_latency(self, X: pd.DataFrame, n_iterations: int = 10000) -> float:
        """
        Measure inference latency per event.
        
        Args:
            X: DataFrame with event features
            n_iterations: Number of timing iterations
            
        Returns:
            Average latency in microseconds
        """
        # Warm-up
        for _ in range(100):
            _ = self.predict(X.iloc[:100])
        
        # Benchmark single-event processing
        times = []
        for _ in range(n_iterations):
            event = X.iloc[np.random.randint(0, len(X))]
            
            start = time.perf_counter()
            _ = self.predict(pd.DataFrame([event]))
            end = time.perf_counter()
            
            times.append((end - start) * 1e6)  # Convert to microseconds
        
        return np.mean(times)
    
    def optimize_cuts(self, X: pd.DataFrame, y: np.ndarray) -> Dict:
        """
        Optimize cut values to maximize signal efficiency.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Dictionary with optimized cut values
        """
        from scipy.optimize import differential_evolution
        
        def objective(params):
            """Negative signal efficiency (to minimize)."""
            pt_cut, chi2_cut = params
            
            mask = (
                (X['pt'] >= pt_cut) &
                (np.abs(X['eta']) <= self.eta_max) &
                (X['chi2'] <= chi2_cut) &
                (X['n_hits'] >= self.min_hits) &
                (X['isolation'] <= self.isolation_max)
            )
            
            predictions = mask.astype(int)
            
            # Calculate signal efficiency and background rate
            signal_mask = y == 1
            background_mask = y == 0
            
            signal_eff = predictions[signal_mask].sum() / signal_mask.sum()
            bkg_rate = predictions[background_mask].sum() / background_mask.sum()
            
            # Penalize high background rates
            if bkg_rate > 0.2:
                return 1.0  # Bad configuration
            
            return -signal_eff  # Maximize efficiency
        
        # Optimize pT and chi2 cuts
        bounds = [(15, 30), (5, 15)]  # pT range, chi2 range
        
        result = differential_evolution(
            objective,
            bounds,
            seed=42,
            maxiter=100,
            workers=1
        )
        
        optimized_cuts = {
            'pt_threshold': result.x[0],
            'chi2_max': result.x[1],
            'eta_max': self.eta_max,
            'min_hits': self.min_hits,
            'isolation_max': self.isolation_max
        }
        
        return optimized_cuts
    
    def get_cut_flow(self, X: pd.DataFrame) -> Dict:
        """
        Analyze event reduction at each cut stage.
        
        Args:
            X: DataFrame with event features
            
        Returns:
            Dictionary with cut flow statistics
        """
        n_total = len(X)
        
        # Apply cuts sequentially
        mask_pt = X['pt'] >= self.pt_threshold
        mask_eta = np.abs(X['eta']) <= self.eta_max
        mask_chi2 = X['chi2'] <= self.chi2_max
        mask_hits = X['n_hits'] >= self.min_hits
        mask_iso = X['isolation'] <= self.isolation_max
        
        # Cumulative cuts
        after_pt = mask_pt.sum()
        after_eta = (mask_pt & mask_eta).sum()
        after_chi2 = (mask_pt & mask_eta & mask_chi2).sum()
        after_hits = (mask_pt & mask_eta & mask_chi2 & mask_hits).sum()
        after_all = (mask_pt & mask_eta & mask_chi2 & mask_hits & mask_iso).sum()
        
        cut_flow = {
            'Initial': n_total,
            'After pT cut': after_pt,
            'After eta cut': after_eta,
            'After chi2 cut': after_chi2,
            'After hits cut': after_hits,
            'After isolation cut': after_all,
            'Final efficiency': after_all / n_total
        }
        
        return cut_flow
    
    def __repr__(self) -> str:
        """String representation of trigger cuts."""
        return (
            f"CutBasedTrigger(\n"
            f"  pT > {self.pt_threshold} GeV\n"
            f"  |η| < {self.eta_max}\n"
            f"  χ² < {self.chi2_max}\n"
            f"  hits ≥ {self.min_hits}\n"
            f"  isolation < {self.isolation_max}\n"
            f")"
        )