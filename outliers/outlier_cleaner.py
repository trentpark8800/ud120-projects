import numpy as np


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    ten_percent_of_data = int(len(ages) * 0.1)
    
    for index in range(0, len(ages)):
        data_tup = tuple(
            (ages[index],
            net_worths[index],
            abs(net_worths[index] - predictions[index]))
        )

        cleaned_data.append(data_tup)
    
    cleaned_data = sorted(cleaned_data, key=lambda x: x[2])

    return cleaned_data[ :-1 * ten_percent_of_data]
