import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TradingStrategy:
    def __init__(self, build_threshold, liquidate_threshold):
        self.build_threshold = build_threshold
        self.liquidate_threshold = liquidate_threshold

    def apply_strategy(self, data):
        data['position'] = 0
        for i in range(1, len(data)):
            if data.loc[i-1, 'alpha'] < self.build_threshold and data.loc[i, 'alpha'] >= self.build_threshold:
                data.loc[i, 'position'] = 1
            elif data.loc[i-1, 'alpha'] >= self.liquidate_threshold and data.loc[i, 'alpha'] < self.liquidate_threshold:
                data.loc[i, 'position'] = 0
            elif data.loc[i-1, 'alpha'] > -self.build_threshold and data.loc[i, 'alpha'] <= -self.build_threshold:
                data.loc[i, 'position'] = -1
            elif data.loc[i-1, 'alpha'] <= -self.liquidate_threshold and data.loc[i, 'alpha'] > -self.liquidate_threshold:
                data.loc[i, 'position'] = 0
            else:
                data.loc[i, 'position'] = data.loc[i-1, 'position']
        return data

class BacktestEngine:
    def __init__(self, strategy, data):
        self.strategy = strategy
        self.data = data

    def run_backtest(self):
        data = self.strategy.apply_strategy(self.data.copy())
        data['returns'] = data['price'].pct_change().fillna(0)
        data['pnl'] = data['position'].shift(1) * data['returns']
        data['cumulative_pnl'] = data['pnl'].cumsum()
        return data

def optimize_thresholds(data):
    # Simple optimization by trying different thresholds
    best_pnl = -np.inf
    best_build = 0
    best_liquidate = 0
    for build in np.arange(0.1, 1.0, 0.1):
        for liquidate in np.arange(0.1, 1.0, 0.1):
            strategy = TradingStrategy(build, liquidate)
            engine = BacktestEngine(strategy, data)
            results = engine.run_backtest()
            final_pnl = results['cumulative_pnl'].iloc[-1]
            if final_pnl > best_pnl:
                best_pnl = final_pnl
                best_build = build
                best_liquidate = liquidate
    return best_build, best_liquidate

def visualize_and_save_backtest_results(data, price_position_path='backtest_results_price_position.png', pnl_path='backtest_results_cumulative_pnl.png'):
    plt.figure(figsize=(14, 7))
    
    # Plot price and positions
    plt.subplot(2, 1, 1)
    plt.plot(data['price'], label='Price')
    plt.plot(data.index, data['position'], label='Position', linestyle='--')
    plt.legend()
    plt.title('Asset Price and Position Over Time')
    plt.xlabel('Time')
    plt.ylabel('Price / Position')
    plt.savefig(price_position_path)
    
    # Plot cumulative PnL
    plt.subplot(2, 1, 2)
    plt.plot(data['cumulative_pnl'], label='Cumulative PnL', color='green')
    plt.legend()
    plt.title('Cumulative PnL Over Time')
    plt.xlabel('Time')
    plt.ylabel('Cumulative PnL')
    plt.savefig(pnl_path)
    
    plt.tight_layout()
    plt.show()

# Load data
data = pd.read_csv('asset_1.csv')

# Optimize thresholds
build_threshold, liquidate_threshold = optimize_thresholds(data)

# Instantiate and use the classes with optimized thresholds
strategy = TradingStrategy(build_threshold=build_threshold, liquidate_threshold=liquidate_threshold)
engine = BacktestEngine(strategy, data)
backtest_results = engine.run_backtest()

# Save the backtest results
backtest_results.to_csv('backtest_results.csv', index=False)

# Visualize and save the backtest results
visualize_and_save_backtest_results(backtest_results)
