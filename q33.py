from math import sqrt, log,exp,pi
from scipy.stats import norm
import pandas as pd
import numpy as np

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
            if price >= S * exp(-q * T) - K * exp(-r * T) and price < S * exp(-q * T):
                return True
            else:
                return False     
        if option_type == 'put':
            if price >= K * exp(-r * T) - S * exp(-q * T) and price < K * exp(-r * T):
                return True
            else:
                return False
    
    def check_put_call_parity(self, S, K, T, t, r, q, call_price, put_price):
        lhs = call_price - put_price
        rhs = S * exp(-q * (T - t)) - K * exp(-r * (T - t))
        if lhs == rhs:
            return 0
        else:
            return lhs - rhs
        


class ArbitrageOpportunities:
    def __init__(self, bs, transaction_cost=3.3):
        self.bs = bs
        self.transaction_cost = transaction_cost

    def hold_latest_instrument_data(self, latest_prices, data):
        strike = data['Strike']
        option_type = data['OptionType']
        bid = data['Bid1']
        ask = data['Ask1']
        if strike not in latest_prices:
            latest_prices[strike] = {}
        latest_prices[strike][option_type + 'BidPrice'] = bid
        latest_prices[strike][option_type + 'AskPrice'] = ask
        return latest_prices

    def put_call_arbitrage(self, option_price, etf_bid_price, etf_ask_price, arbitrage_df, time):
        C = option_price.get('CBidPrice')
        P = option_price.get('PAskPrice')
        S = etf_bid_price
        K = option_price['Strike']
        if pd.isna(C) or pd.isna(P) or pd.isna(S):
            return arbitrage_df

        put_call_parity = self.bs.check_put_call_parity(S, K, T, T0, R, Q, C, P)
        if put_call_parity != 0:
            if put_call_parity < 0:
                if put_call_parity * 10000 < -self.transaction_cost:
                    arbitrage_df.loc[len(arbitrage_df)] = [C, P, K, S, time, 'Put-Call Parity Short sell put, Short-sell underlying, Buy Call, Buy Risk Free Bond', "With Transaction Fee", -put_call_parity * 10000 - self.transaction_cost]
                arbitrage_df.loc[len(arbitrage_df)] = [C, P, K, S, time, 'Put-Call Parity Short sell put, Short-sell underlying, Buy Call, Buy Risk Free Bond', "Without Transaction Fee", -put_call_parity * 10000]
            elif put_call_parity > 0:
                if put_call_parity * 10000 > self.transaction_cost:
                    arbitrage_df.loc[len(arbitrage_df)] = [C, P, K, S, time, 'Put-Call Parity Short sell call, Short-sell bond, Buy Put, Buy Underlying', "With Transaction Fee", put_call_parity * 10000 - self.transaction_cost]
                arbitrage_df.loc[len(arbitrage_df)] = [C, P, K, S, time, 'Put-Call Parity Short sell call, Short-sell bond, Buy Put, Buy Underlying',"Without Transaction Fee", put_call_parity * 10000]
        return arbitrage_df

    def call_option_bound_arbitrage(self, option_price, etf_bid_price, arbitrage_df, time):
        C = option_price.get('CAskPrice')
        S = etf_bid_price
        K = option_price['Strike']
        if pd.isna(C) or pd.isna(S):
            return arbitrage_df

        portfolio_value = S * exp(-Q * T) - K * exp(-R * T) - C
        if portfolio_value > 0 and portfolio_value * 10000 > self.transaction_cost:
            arbitrage_df.loc[len(arbitrage_df)] = [C, np.nan, K, S, time, 'Call Bound', 1, portfolio_value * 10000 - self.transaction_cost]
        return arbitrage_df

    def put_option_bound_arbitrage(self, option_price, etf_ask_price, arbitrage_df, time):
        P = option_price.get('PAskPrice')
        S = etf_ask_price
        K = option_price['Strike']
        if pd.isna(P) or pd.isna(S):
            return arbitrage_df

        portfolio_value = K * exp(-R * T) - S * exp(-Q * T) - P
        if portfolio_value > 0 and portfolio_value * 10000 > self.transaction_cost:
            arbitrage_df.loc[len(arbitrage_df)] = [np.nan, P, K, S, time, 'Put Bound', 1, portfolio_value * 10000 - self.transaction_cost]
        return arbitrage_df

    def vertical_spread_arbitrage(self, strike, latest_prices, arbitrage_df, time):
        for K1, prices in latest_prices.items():
            if K1 == strike:
                continue
            C1 = prices.get('CAskPrice')
            C2 = latest_prices[strike].get('CBidPrice')
            if pd.isna(C1) or pd.isna(C2):
                continue

            portfolio_value = C2 - C1
            if portfolio_value > 0 and portfolio_value * 10000 > self.transaction_cost:
                arbitrage_df.loc[len(arbitrage_df)] = [(C1, C2), np.nan, (K1, strike), np.nan, time, 'Vertical Spread', 1, portfolio_value * 10000 - self.transaction_cost]
        return arbitrage_df

    def non_negative_butterfly_spread_arbitrage(self, strike, latest_prices, arbitrage_df, time):
        K1 = strike
        C1 = latest_prices[strike].get('CAskPrice')
        for K2, prices2 in latest_prices.items():
            if K2 <= K1:
                continue
            C2 = prices2.get('CBidPrice')
            for K3, prices3 in latest_prices.items():
                if K3 <= K2:
                    continue
                C3 = prices3.get('CAskPrice')
                a_b = (K3 - K2) / (K3 - K1)
                portfolio_value = C2 - a_b * C1 - (1 - a_b) * C3
                if portfolio_value > 0 and portfolio_value * 10000 > self.transaction_cost:
                    arbitrage_df.loc[len(arbitrage_df)] = [(C1, C2, C3), np.nan, (K1, K2, K3), np.nan, time, 'Non Negative Butterfly', 1, portfolio_value * 10000 - self.transaction_cost]
        return arbitrage_df

def main():
    instruments = pd.read_csv('instruments.csv')
    marketdata = pd.read_csv('marketdata.csv')
    merged_df = instruments.merge(marketdata, on='Symbol')
    marketdata = merged_df.sort_values('LocalTime').reset_index(drop=True)
    etf_bid_price = np.nan
    etf_ask_price = np.nan
    latest_prices = {}
    arbitrage_df = pd.DataFrame(columns=['ETFPrice', 'CallPrice', 'PutPrice', 'StrikePrice', 'LocalTime', 'Type', 'TransactionArbitrage', 'Profit'])

    bs = BlackScholesModel()
    arbitrage = ArbitrageOpportunities(bs)

    for _, data in marketdata.iterrows():
        if data['Type'] == 'Option':
            latest_prices = arbitrage.hold_latest_instrument_data(latest_prices, data)

            arbitrage_df = arbitrage.put_call_arbitrage(latest_prices[data['Strike']], etf_bid_price, etf_ask_price, arbitrage_df, data['LocalTime'])
            arbitrage_df = arbitrage.call_option_bound_arbitrage(latest_prices[data['Strike']], etf_bid_price, arbitrage_df, data['LocalTime'])
            arbitrage_df = arbitrage.put_option_bound_arbitrage(latest_prices[data['Strike']], etf_ask_price, arbitrage_df, data['LocalTime'])
            if data['OptionType'] == 'C':
                arbitrage_df = arbitrage.vertical_spread_arbitrage(data['Strike'], latest_prices, arbitrage_df, data['LocalTime'])
                arbitrage_df = arbitrage.non_negative_butterfly_spread_arbitrage(data['Strike'], latest_prices, arbitrage_df, data['LocalTime'])
        else:
            etf_bid_price = data['Bid1']
            etf_ask_price = data['Ask1']

    arbitrage_df.to_csv('arbitrage_opportunities.csv')

if __name__ == "__main__":
    main()
