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
            C = S * exp(-q * (T - t)) * norm.cdf(d1) - K * exp(-r * (T - t)) * norm.cdf(d2)
            return C
        
        if option_type == "put":
            P = K * exp(-r * (T - t)) * norm.cdf(-d2) - S * exp(-q * (T - t)) * norm.cdf(-d1)
            return P

    def calculate_d1(self, S, K, T, t, sigma, r, q):
        d1 = ((log(S / K) + (r - q) * (T - t)) / (sigma * sqrt(T - t))) + 0.5 * sigma * sqrt(T - t)
        return d1

    def calculate_d2(self, d1, sigma, T, t):
        d2 = d1 - (sigma * sqrt(T - t))
        return d2

    def calculate_vega(self, S, K, T, t, r, sigma, q):
        d1 = self.calculate_d1(S, K, T, t, sigma, r, q)
        vega = S * exp(-q * (T - t)) * norm.pdf(d1) * sqrt(T - t)
        return vega

    def verify_bounds(self, S, K, T, t, r, q, price, option_type):
        if option_type == 'call':
            if price > S * exp(-q * T) - K * exp(-r * T) and price < S * exp(-q * T):
                return True
            else:
                return False
        if option_type == 'put':
            if price > K * exp(-r * T) - S * exp(-q * T) and price < K * exp(-r * T):
                return True
            else:
                return False


def main():
    #run this code to test the output of the class and implied volatility function
    bs_model = BlackScholesModel()
    risk_free_rate = 0.03
    spot_price = 2
    strike_price = 2
    time_to_maturity = 3
    current_time = 0
    volatility = 0.4
    repo_rate = 0
    true_call_price = bs_model.calculate_option_price(spot_price, strike_price, time_to_maturity, current_time, volatility, risk_free_rate, repo_rate, 'call')
    implied_volatility = calculate_implied_volatility(spot_price, strike_price, time_to_maturity, current_time, risk_free_rate, repo_rate, true_call_price, 'call')
    print(f"Actual Volatility is {volatility}")
    print(f"Implied Volatility from Calculation is {implied_volatility}")
    print(f"The difference between the actual and implied volatility is {volatility - implied_volatility}")

if __name__ == "__main__":
    main()
