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
        """
        Analyzes put-call parity arbitrage opportunities and updates the provided DataFrame with the results.
        Parameters:
        strike (float): The strike price of the options.
        option_price (dict): A dictionary containing the bid and ask prices for call and put options.
        etf_bid_price (float): The bid price of the ETF.
        etf_ask_price (float): The ask price of the ETF.
        arbitrage_df (pd.DataFrame): A DataFrame to store the arbitrage opportunities.
        time (datetime): The time at which the arbitrage opportunity is being evaluated.
        Returns:
        pd.DataFrame: Updated DataFrame with arbitrage opportunities.
        """
        
        C = option_price.get('CBidPrice')
        P = option_price.get('PAskPrice')
        S = etf_ask_price
        K = strike
        
        if pd.isna(C) or pd.isna(P) or pd.isna(S):
            return arbitrage_df
        
        put_call_parity = self.bs.check_put_call_parity(S, K, T, T0, R, Q, C, P)
        if put_call_parity != 0:
            if put_call_parity > 0:
                if put_call_parity * 10000 > self.transaction_cost:
                    arbitrage_df.loc[len(arbitrage_df)] = [S, C, P, K, time, 'Put Call Parity, Sell Call Buy Put', "With Transaction Fee", put_call_parity * 10000 - self.transaction_cost]
                arbitrage_df.loc[len(arbitrage_df)] = [S, C, P, K, time, 'Put Call Parity, Sell Call Buy Put', "Without Transaction Fee", put_call_parity * 10000]
            

        C = option_price.get('CAskPrice')
        P = option_price.get('PBidPrice')
        S = etf_bid_price
        K = strike

        if pd.isna(C) or pd.isna(P) or pd.isna(S):
            return arbitrage_df

        put_call_parity = self.bs.check_put_call_parity(S, K, T, T0, R, Q, C, P)
        if put_call_parity < 0:
            if put_call_parity * 10000 < -1*self.transaction_cost:
                arbitrage_df.loc[len(arbitrage_df)] = [S, C, P, K, time, 'Put Call Parity, Sell Put Buy Call', "With Transaction Fee", put_call_parity*-10000 - self.transaction_cost]
            arbitrage_df.loc[len(arbitrage_df)] = [S, C, P, K, time, 'Put Call Parity, Sell Put Buy Call',"Without Transaction Fee", put_call_parity * -10000]
        
        return arbitrage_df

    def call_option_bound_arbitrage(self, strike, option_price, etf_bid_price, etf_ask_price,arbitrage_df, time):
        def call_option_bound_arbitrage(self, strike, option_price, etf_bid_price, etf_ask_price, arbitrage_df, time):
            """
            Evaluates arbitrage opportunities for call options based on their lower and upper bounds.

            Parameters:
            strike (float): The strike price of the call option.
            option_price (dict): A dictionary containing the bid and ask prices of the call option.
            etf_bid_price (float): The bid price of the underlying ETF.
            etf_ask_price (float): The ask price of the underlying ETF.
            arbitrage_df (pd.DataFrame): A DataFrame to store arbitrage opportunities.
            time (datetime): The current time or timestamp of the evaluation.

            Returns:
            pd.DataFrame: Updated DataFrame with identified arbitrage opportunities.

            Notes:
            - The function checks if the call option is undervalued by comparing it to the lower bound.
            - If the call option is undervalued then you and the potential profit exceeds transaction costs, it records the opportunity.
            - The function also checks if the call option is overvalued by comparing it to the upper bound.
            - If the call option is overvalued, it records the opportunity without considering transaction costs.
            """
        
        C = option_price.get('CAskPrice') 
        S = etf_bid_price
        K = strike

        if pd.isna(C) or  pd.isna(S):
            return arbitrage_df

        lower_bound = np.max(S * exp(-Q * T) - K * exp(-R * T), 0)
        if lower_bound>C:
            if (lower_bound-C) * 10000 > self.transaction_cost:
                arbitrage_df.loc[len(arbitrage_df)] = [S, C, np.nan, K, time, 'Lower Call Bound, Long risk free Bond and short Stock and Buy Call', "With Fees", (lower_bound-C) * 10000 - self.transaction_cost]

        C = option_price.get('CBidPrice')
        S = etf_ask_price
        K = strike

        if pd.isna(C) or pd.isna(S):
            return arbitrage_df

        upper_bound = S * exp(-Q * T)
        if C > upper_bound:
            arbitrage_df.loc[len(arbitrage_df)] = [S, C, np.nan, K, time, 'Upper Call Bound, short the call and Long the stock and short risk free bond', "Without Fees", (C - upper_bound) * 10000]

        return arbitrage_df

    def put_option_bound_arbitrage(self, strike, option_price, etf_bid_price, etf_ask_price, arbitrage_df, time):
        """
        Evaluates arbitrage opportunities for put options based on their lower and upper bounds.

        Parameters:
        strike (float): The strike price of the put option.
        option_price (dict): A dictionary containing the bid and ask prices of the put option.
        etf_bid_price (float): The bid price of the underlying ETF.
        etf_ask_price (float): The ask price of the underlying ETF.
        arbitrage_df (pd.DataFrame): A DataFrame to store arbitrage opportunities.
        time (datetime): The current time or timestamp of the evaluation.

        Returns:
        pd.DataFrame: Updated DataFrame with identified arbitrage opportunities.

        Notes:
        - The function checks if the put option is undervalued by comparing it to the lower bound.
        - If the put option is undervalued and the potential profit exceeds transaction costs, it records the opportunity.
        - The function also checks if the put option is overvalued by comparing it to the upper bound.
        - If the put option is overvalued, it records the opportunity without considering transaction costs.
        """
        
        P = option_price.get('PAskPrice')
        S = etf_ask_price
        K = strike

        if pd.isna(P) or pd.isna(S):
            return arbitrage_df
        
        lower_bound = np.max(K * exp(-R * T) - S * exp(-Q * T), 0)

        if lower_bound > P:
            if (lower_bound - P) * 10000 > self.transaction_cost:
                arbitrage_df.loc[len(arbitrage_df)] = [S, np.nan, P, K, time, 'Lower Put Bound, long the stock and put option and short the risk free bond', "With Fees", (lower_bound - P) * 10000 - self.transaction_cost]
                arbitrage_df.loc[len(arbitrage_df)] = [S, np.nan, P, K, time, 'Lower Put Bound', "With Fees", (lower_bound - P) * 10000]
        
        
        P = option_price.get('PBidPrice')
        S = etf_ask_price
        K = strike

        if pd.isna(P) or pd.isna(S):
            return arbitrage_df

        upper_bound = K * exp(-R * T)
        if P > upper_bound:
            arbitrage_df.loc[len(arbitrage_df)] = [S, np.nan, P, K, time, 'Upper Put Bound, Short put and long risk free bond', "Without Fees", (P - upper_bound) * 10000]
        return arbitrage_df

def main():
    instruments = pd.read_csv('instruments.csv')
    marketdata = pd.read_csv('marketdata.csv')
    merged_df = instruments.merge(marketdata, on='Symbol')
    marketdata = merged_df.sort_values('LocalTime').reset_index(drop=True)
    etf_bid_price = np.nan
    etf_ask_price = np.nan
    latest_prices = {}
    arbitrage_df = pd.DataFrame(columns=['ETFPrice', 'CallPrice', 'PutPrice', 'StrikePrice', 'LocalTime', 'Type and Action', 'Transaction Type', 'Profit'])

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
