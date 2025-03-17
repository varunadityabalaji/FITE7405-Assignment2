from math import sqrt, log,exp,pi
from scipy.stats import norm
import pandas as pd
import numpy as np


Q=0.2
R=0.04
T0=0
T=(24-16)/365


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

    def put_call_arbitrage(self, strike, option_price, etf_bid_price, etf_ask_price, arbitrage_df, time):
        
        #Call is overvalued
        C = option_price.get('CBidPrice')
        P = option_price.get('PAskPrice')
        S = etf_ask_price
        K = strike

        if any(pd.isna(x) for x in [C, P, S]):
            return arbitrage_df

        put_call_parity = self.bs.check_put_call_parity(S, K, T, T0, R, Q, C, P)
        if put_call_parity != 0:
            if put_call_parity > 0:
                if put_call_parity * 10000 > self.transaction_cost:
                    arbitrage_df.loc[len(arbitrage_df)] = [S, C, P, K, time, 'Put Call Parity, Sell Call Buy Put', "With Transaction Fee", put_call_parity * 10000 - self.transaction_cost]
                arbitrage_df.loc[len(arbitrage_df)] = [S, C, P, K, time, 'Put Call Parity, Sell Call Buy Put', "Without Transaction Fee", put_call_parity * 10000]
            

        #Put is overvalued
        C = option_price.get('CAskPrice')
        P = option_price.get('PBidPrice')
        S = etf_bid_price
        K = strike

        put_call_parity = self.bs.check_put_call_parity(S, K, T, T0, R, Q, C, P)
        if put_call_parity < 0:
            if put_call_parity * 10000 < -1*self.transaction_cost:
                arbitrage_df.loc[len(arbitrage_df)] = [S, C, P, K, time, 'Put-Call Parity Short sell call, Short-sell bond, Buy Put, Buy Underlying', "With Transaction Fee", put_call_parity*-10000 - self.transaction_cost]
            arbitrage_df.loc[len(arbitrage_df)] = [S, C, P, K, time, 'Put-Call Parity Short sell call, Short-sell bond, Buy Put, Buy Underlying',"Without Transaction Fee", put_call_parity * -10000]
        
        return arbitrage_df

    def call_option_bound_arbitrage(self, strike, option_price, etf_bid_price, etf_ask_price,arbitrage_df, time):
        C = option_price.get('CAskPrice') 
        S = etf_bid_price
        K = strike

        if any(pd.isna(x) for x in [C, S]):
            return arbitrage_df

        lower_bound = np.max(S * exp(-Q * T) - K * exp(-R * T) - C, 0)
        if lower_bound>C:
            if (lower_bound-C) * 10000 > self.transaction_cost:
                arbitrage_df.loc[len(arbitrage_df)] = [S, C, np.nan, K, time, 'Lower Call Bound', "With Fees", (lower_bound-C) * 10000 - self.transaction_cost]

        C = option_price.get('CBidPrice')
        S = etf_ask_price
        K = strike

        upper_bound = S * exp(-Q * T)
        if C > upper_bound:
            arbitrage_df.loc[len(arbitrage_df)] = [S, C, np.nan, K, time, 'Upper Call Bound', "Without Fees", (C - upper_bound) * 10000]

        return arbitrage_df

    def put_option_bound_arbitrage(self, strike, option_price, etf_bid_price, etf_ask_price, arbitrage_df, time):
        P = option_price.get('PAskPrice')
        S = etf_bid_price
        K = strike
        if pd.isna(P) or pd.isna(S):
            return arbitrage_df
        
        lower_bound = np.max(K * exp(-R * T) - S * exp(-Q * T), 0)

        if lower_bound > P:
            if (lower_bound - P) * 10000 > self.transaction_cost:
                arbitrage_df.loc[len(arbitrage_df)] = [S, np.nan, P, K, time, 'Lower Put Bound', "With Fees", (lower_bound - P) * 10000 - self.transaction_cost]
                arbitrage_df.loc[len(arbitrage_df)] = [S, np.nan, P, K, time, 'Lower Put Bound', "With Fees", (lower_bound - P) * 10000]
        
        
        P = option_price.get('PBidPrice')
        S = etf_ask_price
        K = strike

        upper_bound = K * exp(-R * T)
        if P > upper_bound:
            arbitrage_df.loc[len(arbitrage_df)] = [S, np.nan, P, K, time, 'Upper Put Bound', "Without Fees", (P - upper_bound) * 10000]
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
            arbitrage_df = arbitrage.put_call_arbitrage(data['Strike'],latest_prices[data['Strike']], etf_bid_price, etf_ask_price, arbitrage_df, data['LocalTime'])
            arbitrage_df = arbitrage.call_option_bound_arbitrage(data['Strike'],latest_prices[data['Strike']], etf_bid_price, etf_ask_price, arbitrage_df, data['LocalTime'])
            arbitrage_df = arbitrage.put_option_bound_arbitrage(data['Strike'],latest_prices[data['Strike']], etf_ask_price, etf_ask_price, arbitrage_df, data['LocalTime'])
        else:
            etf_bid_price = data['Bid1']
            etf_ask_price = data['Ask1']

    arbitrage_df.to_csv('arbitrage_opportunities.csv')

if __name__ == "__main__":
    main()
