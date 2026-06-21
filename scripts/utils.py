# -*- coding: utf-8 -*-
"""
Created on Tue Oct 29 13:04:53 2024

@author: HP
"""
import numpy as np


# Function to create binary representation for a given range
def create_binary_array(values, start_value, end_value):
    total_size = abs(end_value - start_value) + 1
    binary_array = np.zeros(total_size)  # Initialize an array of zeros
    for value in values:
        if start_value <= value <= end_value:  # Check if value is within the range
            binary_array[value - start_value] = 1  # Set the corresponding index to 1
    return binary_array

def convert_pulse_width(value):
    if '-' in value:
        # Split the two values and take the average
        parts = value.replace(' ms', '').replace(' μs', '').split('-')
        average = (float(parts[0]) + float(parts[1])) / 2
        if 'ms' in value:
            return average * 1000
        elif 'μs' in value:
            return average
    elif 'ms' in value:
        return float(value.replace(' ms', '')) * 1000
    elif 'μs' in value:
        return float(value.replace(' μs', ''))
    else:
        return float(value)
