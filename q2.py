import numpy as np

class CorrelationCoefficientVerifier(object):
    def __init__(self):
        np.random.seed(10)
        self.X = np.random.standard_normal(200)
        self.Y = np.random.standard_normal(200)

    def proof_checker(self):
        p = 0.5
        Z = p * self.X + np.sqrt(1 - p**2) * self.Y
        correlation_coefficient = np.corrcoef(Z, self.X)[0, 1]
        print("The correlation coefficient is:", correlation_coefficient)
        difference = correlation_coefficient - p
        print("The difference between the correlation coefficient and the expected value is: ", difference)

# Example usage:
if __name__ == '__main__':
    verifier = CorrelationCoefficientVerifier()
    verifier.proof_checker()


#generate more samples and show mean and talk about law of large numbers and central limit theorem

