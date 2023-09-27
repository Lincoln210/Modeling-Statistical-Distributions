# SYSTEM IMPORTS
from abc import abstractmethod, ABC     # you need these to make an abstract class in Python
from typing import List, Type, Union    # Python's typing syntax
import numpy as np                      # linear algebra & useful containers

# PYTHON PROJECT IMPORTS


# Types defined in this module
DistributionType: Type = Type["Distribution"]
BinomialDistributionType = Type["BinomialDistribution"]
PoissonDistributionType = Type["PoissonDistribution"]
GaussianDistributionType = Type["GaussianDistribution"]

# an abstract class for an arbitrary distribution
# please don't touch this class
class Distribution(ABC):

    # this is how you make an abstract method in Python
    # all child classes MUST implement this method
    # otherwise we will get an error when the child class
    # is instantiated 
    @abstractmethod 
    def fit(self: DistributionType,
            X: np.ndarray               # input data to fit from
            ) -> DistributionType:
        ... # same as "pass"

    # another abstract method that every child class will have to implement
    # in order to be able to be instantiated
    @abstractmethod
    def prob(self: DistributionType,
             X: np.ndarray
             ) -> np.ndarray:           # return Pr[x] for each point (row) of the data (X)
        ... # same as "pass"

    @abstractmethod
    def parameters(self: DistributionType) -> List[Union[float, np.ndarray]]:
        ... # same as "pass"


# a class for the binomial distribution
# you will need to complete this class
class BinomialDistribution(Distribution):
    def __init__(self: BinomialDistributionType,
                 n: int) -> None:
        # controlled by parameter "p"
        self.p: float = None
        self.n: int = n


    def fit(self: BinomialDistributionType,
            X: np.ndarray               # input data to fit from
            ) -> BinomialDistributionType:
        # TODO: complete me!
        count = 0
        for i in X:
            count += sum(i)
                    
        self.p = count / self.n

        return self # keep this at the end


    def prob(self: BinomialDistributionType,
             X: np.ndarray
             ) -> np.ndarray:
        # TODO: complete me!
        num_successes = np.sum(X, axis=1)
        num_failures = X.shape[1] - num_successes
        probabilities = (self.p ** num_successes) * ((1 - self.p) ** num_failures)
        
        return probabilities.reshape(-1, 1)

    def parameters(self: BinomialDistributionType) -> List[Union[float, np.ndarray]]:
        return [self.n, self.p]


class GaussianDistribution(Distribution):
    def __init__(self: GaussianDistributionType) -> None:
        # controlled by parameters mu and var
        self.mu: float = None
        self.var: float = None

    def fit(self: GaussianDistributionType,
            X: np.ndarray                   # input data to fit from
                                            # this will be a bunch of integer samples stored in a column vector
            ) -> GaussianDistributionType:

        # TODO: complete me!
        self.mu = np.mean(X)
        
        squared_differences = [(x - self.mu) ** 2 for x in X]
        self.var = sum(squared_differences) / len(X)
        return self

    def prob(self: GaussianDistributionType,
             X: np.ndarray                  # this will be a column vector where every element is a float
             ) -> np.ndarray:
        # TODO: complete me!
        coefficient = 1 / np.sqrt(2 * np.pi * self.var)
        exponent = -((X - self.mu) ** 2) / (2 * self.var)
        probabilities = coefficient * np.exp(exponent)
        
        return probabilities.reshape(-1, 1)

    def parameters(self: GaussianDistributionType) -> List[Union[float, np.ndarray]]:
        return [self.mu, self.var]


# a class for the poisson distribution
# you will need to complete this class
class PoissonDistribution(Distribution):
    def __init__(self: PoissonDistributionType) -> None:
        # controlled by parameter "lambda"
        self.lamb: float = None

    def fit(self: PoissonDistributionType,
            X: np.ndarray               # input data to fit from
            ) -> PoissonDistributionType:
        # TODO: complete me!
        self.lamb = sum(X) / len(X)
        return self # keep this at the end

    def prob(self: PoissonDistributionType,
             X: np.ndarray
             ) -> np.ndarray:
        # TODO: complete me!
        event_counts = np.sum(X, axis=1)
        
        factorials = np.vectorize(np.math.factorial)(event_counts)
        poisson_probabilities = ((self.lamb ** event_counts) * np.exp(-self.lamb)) / factorials
        
        return poisson_probabilities.reshape(-1, 1)

    def parameters(self: PoissonDistributionType) -> List[Union[float, np.ndarray]]:
        return [self.lamb]