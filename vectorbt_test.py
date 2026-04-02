import numpy as np
import pandas as pd
import vectorbt as vbt

# Check vectorbt behavior when we have fewer signals (our new conservative model)

prices = pd.Series(np.linspace(100, 200, 100)) # Prices go up constantly
# Buy once, but exit immediately the next day, repeatedly checking behavior
signals = pd.Series([0]*100)
signals[10] = 1
signals[11] = 0

entries = signals == 1
exits = signals == 0

pf = vbt.Portfolio.from_signals(
    close=prices,
    entries=entries,
    exits=exits,
    fees=0.000,
    init_cash=10000,
    freq='1D'
)
print("Return strategy:", pf.total_return())
print("Return B&H:", vbt.Portfolio.from_holding(prices, init_cash=10000).total_return())
