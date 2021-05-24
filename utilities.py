"""
Utilities for E-Z reader. This file collects all basic functions used in E-Z reader.
"""

import math

def time_familiarity_check(distance, wordlength, frequency, predictability, eccentricity, alpha1=104, alpha2=3.4, alpha3=39):
    """
    Time to calculate L1 (familiarity check).

    :distance: distance (in number of characters) between fixation and first letter of the word
    :wordlength: length of words (in number of characters)
    :frequency: frequency of the word
    :predictability: predictability of the word
    :eccentricity: a free parameter
    :otherparameters: other parameters affecting familiarity check (alpha1, alpha2, alpha3)
    return: time of familiarity check in ms
    """
    tL1 = alpha1 - alpha2*math.log(frequency) - alpha3*predictability
    tL1 = tL1 * pow (eccentricity, (distance+(wordlength-1)/2)) #adjust tL1 by eccentricity and distance to the middle point of the word
    return tL1

def time_lexical_access(frequency, predictability, delta, alpha1=104, alpha2=3.4, alpha3=39):
    """
    Time to calculate L2 (lexical access).

    :distance: distance (in number of characters) between fixation and first letter of the word
    :wordlength: length of words (in number of characters)
    :frequency: frequency of the word
    :predictability: predictability of the word
    :eccentricity: a free parameter
    :otherparameters: other parameters affecting familiarity check (alpha1, alpha2, alpha3)
    return: time of familiarity check in ms
    """
    tL2 = delta*(alpha1 - alpha2*math.log(frequency) - alpha3*predictability)
    return tL2

if __name__ == "__main__":
    #examples how to run functions
    tL1 = time_familiarity_check(3, 4, 3e05, 0.2, 1.15)
    tL2 = time_lexical_access(1e05, 0.2, 0.34)
    print(tL1)
    print(tL2)

