"""
Event data generation for trigger simulation.

Generates synthetic muon collision events with realistic kinematic distributions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


class EventGenerator:
    """Generate synthetic collision events for trigger simulation."""
    
    def __init__(self, config: Dict):
        """
        Initialize event generator.
        
        Args:
            config: Configuration dictionary with data parameters
        """
        self.config = config
        self.rng = np.random.RandomState(config.get('random_seed', 42))
    
    def generate_signal_events(self, n_events: int) -> pd.DataFrame:
        """
        Generate signal events (real muons).
        
        Args:
            n_events: Number of events to generate
            
        Returns:
            DataFrame with event features
        """
        signal_cfg = self.config['signal']
        
        # Generate pT with physics-motivated distribution (falling spectrum)
        pt = self._generate_pt_spectrum(
            n_events,
            signal_cfg['pt_mean'],
            signal_cfg['pt_std'],
            signal_cfg['pt_min'],
            signal_cfg['pt_max']
        )
        
        # Generate eta (pseudorapidity) - uniform in detector acceptance
        eta = self.rng.uniform(
            -signal_cfg['eta_max'],
            signal_cfg['eta_max'],
            n_events
        )
        
        # Generate phi (azimuthal angle) - uniform
        phi = self.rng.uniform(-np.pi, np.pi, n_events)
        
        # Generate detector quality variables
        # Signal has good detector hits
        n_hits = self.rng.poisson(15, n_events).clip(8, 25)
        
        # Signal has good track fit (low chi2)
        chi2 = self.rng.gamma(2, 1.5, n_events).clip(0.1, 20)
        
        # Signal muons are isolated (low nearby energy)
        isolation = self.rng.gamma(1, 0.15, n_events).clip(0, 1)
        
        # Pile-up: number of primary vertices
        n_vertices = self.rng.poisson(35, n_events).clip(1, 80)
        
        df = pd.DataFrame({
            'pt': pt,
            'eta': eta,
            'phi': phi,
            'n_hits': n_hits,
            'chi2': chi2,
            'isolation': isolation,
            'n_vertices': n_vertices,
            'label': 1  # Signal
        })
        
        return df
    
    def generate_background_events(self, n_events: int) -> pd.DataFrame:
        """
        Generate background events (fake muons, mis-reconstructed tracks).
        
        Args:
            n_events: Number of events to generate
            
        Returns:
            DataFrame with event features
        """
        bkg_cfg = self.config['background']
        
        # Background has softer pT spectrum
        pt = self._generate_pt_spectrum(
            n_events,
            bkg_cfg['pt_mean'],
            bkg_cfg['pt_std'],
            bkg_cfg['pt_min'],
            bkg_cfg['pt_max']
        )
        
        # Background more forward (larger |eta|)
        eta = self.rng.uniform(
            -bkg_cfg['eta_max'],
            bkg_cfg['eta_max'],
            n_events
        )
        eta = eta + self.rng.normal(0, 0.5, n_events)  # Push toward edges
        eta = eta.clip(-bkg_cfg['eta_max'], bkg_cfg['eta_max'])
        
        phi = self.rng.uniform(-np.pi, np.pi, n_events)
        
        # Background has fewer/worse detector hits
        n_hits = self.rng.poisson(10, n_events).clip(4, 20)
        
        # Background has worse track fit (higher chi2)
        chi2 = self.rng.gamma(4, 2.5, n_events).clip(0.1, 50)
        
        # Background less isolated (more nearby energy)
        isolation = self.rng.gamma(2, 0.25, n_events).clip(0, 1)
        
        n_vertices = self.rng.poisson(35, n_events).clip(1, 80)
        
        df = pd.DataFrame({
            'pt': pt,
            'eta': eta,
            'phi': phi,
            'n_hits': n_hits,
            'chi2': chi2,
            'isolation': isolation,
            'n_vertices': n_vertices,
            'label': 0  # Background
        })
        
        return df
    
    def _generate_pt_spectrum(
        self,
        n_events: int,
        mean: float,
        std: float,
        pt_min: float,
        pt_max: float
    ) -> np.ndarray:
        """
        Generate realistic pT distribution (falling spectrum).
        
        Particle physics spectra typically fall as ~1/pT^n
        """
        # Use exponential with gaussian smearing for realistic spectrum
        scale = mean / 2
        pt = self.rng.exponential(scale, n_events)
        pt = pt + self.rng.normal(0, std/3, n_events)
        pt = pt.clip(pt_min, pt_max)
        
        return pt
    
    def generate_dataset(
        self,
        n_signal: int,
        n_background: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate complete dataset with train/test split.
        
        Args:
            n_signal: Number of signal events
            n_background: Number of background events
            
        Returns:
            Tuple of (full_data, train_data, test_data)
        """
        # Generate events
        signal = self.generate_signal_events(n_signal)
        background = self.generate_background_events(n_background)
        
        # Combine and shuffle
        data = pd.concat([signal, background], ignore_index=True)
        data = data.sample(frac=1, random_state=self.config.get('random_seed', 42))
        data = data.reset_index(drop=True)
        
        # Split train/test (70/30)
        train_size = int(0.7 * len(data))
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        return data, train_data, test_data
    
    def add_noise_and_inefficiencies(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add detector noise and inefficiencies to make data more realistic.
        
        Args:
            data: Event DataFrame
            
        Returns:
            Modified DataFrame with noise
        """
        data = data.copy()
        
        # Add measurement resolution effects
        data['pt'] = data['pt'] * (1 + self.rng.normal(0, 0.02, len(data)))
        data['eta'] = data['eta'] + self.rng.normal(0, 0.01, len(data))
        data['phi'] = data['phi'] + self.rng.normal(0, 0.01, len(data))
        
        # Clip to physical ranges
        data['pt'] = data['pt'].clip(0, None)
        data['eta'] = data['eta'].clip(-2.5, 2.5)
        data['phi'] = data['phi'].clip(-np.pi, np.pi)
        
        # Occasional detector dead channels (set hits to 0)
        dead_channel_mask = self.rng.random(len(data)) < 0.02
        data.loc[dead_channel_mask, 'n_hits'] = data.loc[dead_channel_mask, 'n_hits'] * 0.5
        
        return data