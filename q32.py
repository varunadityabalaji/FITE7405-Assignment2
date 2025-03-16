from math import sqrt, log,exp,pi
from scipy.stats import norm
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from math import sqrt, log,exp,pi
from scipy.stats import norm
import pandas as pd
import numpy as np

def calculate_implied_volatility(S, K, T, t, r, q, C_true, option_type):
    # Newton Raphson Method to calculate Implied Volatility
    sigma_hat = sqrt(2 * abs((log(S / K) + (r - q) * (T - t)) / (T - t)))
    tol = 1e-8  # Tolerance
    n_max = 1000  # Number of Iterations
    sigma_diff = 1
    n = 1
    sigma = sigma_hat
    bs_model = BlackScholesModel()  # Initialize an Option object
    while sigma_diff >= tol and n_max > n:
        if option_type == 'call':
            C = bs_model.calculate_option_price(S, K, T, t, sigma, r, q, 'call')
        else:
            C = bs_model.calculate_option_price(S, K, T, t, sigma, r, q, 'put')
        d1 = bs_model.calculate_d1(S, K, T, t, sigma, r, q)
        C_vega = bs_model.calculate_vega(S, K, T, t, r, sigma, q)
        increment = (C - C_true) / C_vega
        sigma = sigma - increment
        n = n + 1
        sigma_diff = abs(increment)
    return sigma


class BlackScholesModel:
    def calculate_option_price(self, S, K, T, t, sigma, r, q, option_type):
        N = norm.cdf
        d1 = self.calculate_d1(S, K, T, t, sigma, r, q)
        d2 = self.calculate_d2(d1, sigma, T, t)
        if option_type == "call":
            C = S * exp(-q * (T - t)) * N(d1) - K * exp(-r * (T - t)) * N(d2)
            return C
            
        if option_type == "put":
            P = K * exp(-r * (T - t)) * N(-d2) - S * exp(-q * (T - t)) * N(-d1)
            return P


    def calculate_d1(self, S, K, T, t, sigma, r, q):
        d1 = ((log(S / K) + (r - q) * (T - t)) / (sigma * sqrt(T - t))) + 0.5 * sigma * sqrt(T - t)
        return d1

    def calculate_d2(self, d1, sigma, T, t):
        d2 = d1 - (sigma * sqrt(T - t))
        return d2

    def calculate_vega(self, S, K, T, t, r, sigma, q):
        d1 = self.calculate_d1(S, K, T, t, sigma, r, q)
        vega = S * exp(-q * (T - t)) * sqrt(T - t) * exp(-0.5 * d1 * sigma ** 2) / sqrt(2 * pi)
        return vega

    def verify_bounds(self, S, K, T, t, r, q, price, option_type):
        if option_type == 'call':
            if price >= S * exp(-q * T) - K * exp(-r * T) and price <= S * exp(-q * T):
                return True
            else:
                return False
        if option_type == 'put':
            if price >= K * exp(-r * T) - S * exp(-q * T) and price <= K * exp(-r * T):
                return True
            else:
                return False


def calculate_implied_volatility_for_data(data, r, q, T, t, end_time):
    #intialise the black scholes class
    bs_model = BlackScholesModel()

    S = (data[(data['Symbol'] == 510050) & (data['LocalTime'] <= end_time)].iloc[-1]['Ask1'] + 
         data[(data['Symbol'] == 510050) & (data['LocalTime'] <= end_time)].iloc[-1]['Bid1']) / 2
    
    strikes = data[(data['Symbol'] != 510050) & (data['LocalTime'] <= end_time)]['Strike'].dropna().unique()
    implied_vol_df = pd.DataFrame(index=strikes, columns=['BidVolP', 'AskVolP', 'BidVolC', 'AskVolC'])
    implied_vol_df.index.name = 'Strike'
    
    #loop through all the options and calculate the implied volatility
    for symbol in data['Symbol'].unique()[:-1]:
        instrument = data[(data['Symbol'] == symbol) & (data['LocalTime'] <= end_time)].dropna().iloc[-1]
        K = instrument['Strike']
        true_bid = instrument['Bid1']
        true_ask = instrument['Ask1']
        option_type = 'call' if instrument['OptionType'] == 'C' else 'put'
        
        if bs_model.verify_bounds(S, K, T, t, r, q, true_bid, option_type):
            bid_vol = calculate_implied_volatility(S, K, T, t, r, q, true_bid, option_type)
            implied_vol_df.loc[K, 'BidVol' + option_type[0].upper()] = bid_vol
        
        else: 
            implied_vol_df.loc[K, 'BidVol' + option_type[0].upper()] = 'NaN'
        
        if bs_model.verify_bounds(S, K, T, t, r, q, true_ask, option_type):
            ask_vol = calculate_implied_volatility(S, K, T, t, r, q, true_ask, option_type)
            implied_vol_df.loc[K, 'AskVol' + option_type[0].upper()] = ask_vol
        else: 
            implied_vol_df.loc[K, 'AskVol' + option_type[0].upper()] = 'NaN'
    
    return implied_vol_df.sort_index()

def plot_implied_volatility(df, time):
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(time)
    axs[0, 0].plot(df.index, df.BidVolP)
    axs[0, 0].set_title("BidVolP")
    axs[1, 0].plot(df.index, df.BidVolC, 'tab:orange')
    axs[1, 0].set_title("BidVolC")
    axs[1, 0].sharex(axs[0, 0])
    axs[0, 1].plot(df.index, df.AskVolP, 'tab:green')
    axs[0, 1].set_title("AskVolP")
    axs[1, 1].plot(df.index, df.AskVolC, 'tab:red')
    axs[1, 1].set_title("AskVolC")
    axs[1, 1].sharex(axs[0, 1])
    fig.tight_layout()
    plt.show()



def main():
    instruments = pd.read_csv('instruments.csv')
    market_data = pd.read_csv('marketdata.csv')
    merged_data = instruments.merge(market_data, on='Symbol')

    r = 0.04
    q = 0.2
    t = 0
    T = (24 - 16) / 365

    end_times = ['2016-Feb-16 09:31', '2016-Feb-16 09:32', '2016-Feb-16 09:33']
    for end_time in end_times:
        df = calculate_implied_volatility_for_data(merged_data, r, q, T, t, end_time)
        df.to_csv(f'{end_time[-2:]}.csv')
        plot_implied_volatility(df, end_time[-5:])

if __name__ == '__main__':
    main()