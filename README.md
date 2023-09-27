# Modeling-Statistical-Distributions
## Overview
This framework provides an efficient way to model statistical distributions, focusing on the Binomial, Gaussian (Normal), and Poisson distributions. By utilizing object-oriented principles, each distribution is represented as a class, ensuring clear separation and ease of expansion for other distributions in the future.
## Features
### 1. Abstract Distribution Class
- A blueprint (Distribution) that all specific distribution classes inherit from.
- Requires any derived class to implement fit, prob, and parameters methods.
### 2. Binomial Distribution
- Estimate the probability of success (parameter 'p').
- Calculate the probability for a set of data points.
- Fetch the parameters n (number of trials) and p (probability of success).
### 3. Gaussian (Normal) Distribution
- Estimate mean and variance from data samples.
- Calculate the probability density for a set of data points.
- Fetch the parameters n (number of trials) and p (probability of success).
### 4. Poisson Distribution
- Estimate the average rate (lambda) from data samples.
- Calculate the probability for a set of data points.
- Fetch the lambda parameter.
