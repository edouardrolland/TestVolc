def calculate_derivative(values, sampling_time):
    derivative_values = []
    for i in range(1, len(values)):
        derivative = (values[i] - values[i-1]) / sampling_time
        derivative_values.append(derivative)
    return derivative_values