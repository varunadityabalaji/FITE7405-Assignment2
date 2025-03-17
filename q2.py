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
        
        # Print mean and variance of X and Y
        print("Mean of X:", np.mean(self.X))
        print("Variance of X:", np.var(self.X))
        print("Mean of Y:", np.mean(self.Y))
        print("Variance of Y:", np.var(self.Y))

# Example usage:
if __name__ == '__main__':
    verifier = CorrelationCoefficientVerifier()
    verifier.proof_checker()


#from central limit theorem and law of large numbers, with more samples the results get better

