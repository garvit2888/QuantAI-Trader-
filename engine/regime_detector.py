import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

class MarketRegimeDetector:
    """
    Identifies market regimes (Bull, Bear, Sideways, Volatile) 
    using a Gaussian Mixture Model (GMM) on Volatility and Momentum.
    """
    def __init__(self, n_regimes=4):
        self.n_regimes = n_regimes
        self.model = GaussianMixture(n_components=n_regimes, random_state=42, n_init=10)
        self.regime_map = {} # Maps cluster ID to human-readable name
        self.is_fitted = False

    def fit(self, df):
        """
        Fits the GMM on the dataframe. 
        Expects columns: 'Volatility_Regime' and 'Market_Trend_Regime'.
        """
        if 'Volatility_Regime' not in df.columns or 'Market_Trend_Regime' not in df.columns:
            return self
            
        # Select data and drop NaNs
        X = df[['Volatility_Regime', 'Market_Trend_Regime']].ffill().dropna()
        if len(X) < 10: # Lowered threshold for fit, but will use full history if possible
            return self
            
        n_clusters = min(self.n_regimes, len(X))
        if n_clusters < self.n_regimes:
            self.model = GaussianMixture(n_components=n_clusters, random_state=42)
            
        self.model.fit(X)
        
        # Determine labels based on cluster characteristics
        centers = self.model.means_ 
        
        roles = []
        for i in range(n_clusters):
            # Fallback if 1D
            vol = centers[i][0] if centers.shape[1] > 0 else 1.0
            trend = centers[i][1] if centers.shape[1] > 1 else 1.0
            roles.append({'id': i, 'vol': vol, 'trend': trend})
            
        # Refined mapping logic (using medians as pivots)
        v_med = np.median(centers[:, 0]) if centers.shape[1] > 0 else 1.0
        t_med = np.median(centers[:, 1]) if centers.shape[1] > 1 else 1.0
        
        for r in roles:
            vol, trend = r['vol'], r['trend']
            if vol > np.percentile(centers[:, 0], 80) if len(centers) > 1 else False:
                name, color = "VOLATILE PIVOT", "#F59E0B"
            elif trend > t_med and vol <= v_med:
                name, color = "STEADY BULL", "#22C55E"
            elif trend < t_med and vol > v_med:
                name, color = "ANGRY BEAR", "#EF4444"
            else:
                name, color = "SIDEWAYS / FLAT", "#3B82F6"
            
            self.regime_map[r['id']] = {"name": name, "color": color}
            
        self.is_fitted = True
        return self

    def predict(self, df):
        """Returns the regime name and color for the last row of the dataframe."""
        if not self.is_fitted:
            return "UNKNOWN", "#9CA3AF"
            
        X = df[['Volatility_Regime', 'Market_Trend_Regime']].iloc[-1:].ffill().fillna(0)
        cluster = self.model.predict(X)[0]
        return self.regime_map[cluster]["name"], self.regime_map[cluster]["color"]

if __name__ == "__main__":
    # Test logic
    data = pd.DataFrame({
        'Volatility_Regime': np.random.normal(1, 0.2, 100),
        'Market_Trend_Regime': np.random.normal(1, 0.1, 100)
    })
    detector = MarketRegimeDetector().fit(data)
    name, color = detector.predict(data)
    print(f"Detected Regime: {name} (Color: {color})")
