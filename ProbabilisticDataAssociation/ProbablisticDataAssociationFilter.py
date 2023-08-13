import numpy as np


class ProbabilisticDataAssociationFilter:
    """Implement the PDA filter under the assumption mentioned"""

    def __init__(self):
        pass

    def predict(self):
        """Predict the future state, measurement and covariance based on a known model"""
        pass

    def validate(self, measurements):
        """Create validation region based on the measurement"""
        # Todo: create measurements as list or set and create validation range
        pass

    def evaluate_association_probability(self):
        pass

    def update(self):
        """Update the state and covariance based on the measurements and innovation"""
        pass