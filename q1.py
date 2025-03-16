import scipy
import numpy as np

class CallOptionValuation(object):
    def __init__(self,stock_price:float,rate:float,strike:float,expiry:float,volatility:float):
        self.stock_price = stock_price
        self.rate = rate
        self.strike = strike
        self.expiry = expiry
        self.volatility = volatility

    #used to calcuate d1 
    def calculate_d1(self):
        d1 = (np.log(self.stock_price / self.strike) + (self.rate + 0.5 * self.volatility ** 2) * self.expiry) / (self.volatility * np.sqrt(self.expiry))
        return d1

    #used to calculate d2
    def calculate_d2(self):
        d2 = self.calculate_d1() - self.volatility * np.sqrt(self.expiry)
        return d2
    
    #used to calculate the call option price
    def calculate_call_price(self):
        d1 = self.calculate_d1()
        d2 = self.calculate_d2()
        call_price = self.stock_price * scipy.stats.norm.cdf(d1) - self.strike * np.exp(-self.rate * self.expiry) * scipy.stats.norm.cdf(d2)
        return call_price


class PutOptionValuation(object):
    def __init__(self, stock_price: float, rate: float, strike: float, expiry: float, volatility: float):
        self.stock_price = stock_price
        self.rate = rate
        self.strike = strike
        self.expiry = expiry
        self.volatility = volatility

    # used to calculate d1
    def calculate_d1(self):
        d1 = (np.log(self.stock_price / self.strike) + (self.rate + 0.5 * self.volatility ** 2) * self.expiry) / (self.volatility * np.sqrt(self.expiry))
        return d1

    # used to calculate d2
    def calculate_d2(self):
        d2 = self.calculate_d1() - self.volatility * np.sqrt(self.expiry)
        return d2

    # used to calculate the put option price
    def calculate_put_price(self):
        d1 = self.calculate_d1()
        d2 = self.calculate_d2()
        put_price = self.strike * np.exp(-self.rate * self.expiry) * scipy.stats.norm.cdf(-d2) - self.stock_price * scipy.stats.norm.cdf(-d1)
        return put_price


if __name__ == '__main__':
    options = [
        (100, 100, 1.0, 0.3, 0.03),
        (100, 110, 1.0, 0.3, 0.03),
        (100, 100, 1.5, 0.3, 0.03),
        (100, 100, 1.0, 0.4, 0.03),
        (100, 100, 1.0, 0.3, 0.05),
        (110, 100, 1.0, 0.3, 0.03)
    ]

    for S, K, T, sigma, r in options:
        call_option = CallOptionValuation(S, r, K, T, sigma)
        put_option = PutOptionValuation(S, r, K, T, sigma)
        print(f"Call Option Price for S={S}, K={K}, T={T}, sigma={sigma}, r={r}: {call_option.calculate_call_price()}")
        print(f"Put Option Price for S={S}, K={K}, T={T}, sigma={sigma}, r={r}: {put_option.calculate_put_price()}")
