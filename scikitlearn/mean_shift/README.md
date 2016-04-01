# mean_shift.py
Generates random data and uses the mean shift algorithm to cluster it. Prints the coordinates of each centers computed, and prints the graph with each cluster in a different color.

# sequential_mean_shift.py
Try to implement mean_shift with sequential data. It doesn't work because the function fit() reset each time it is called. On each new graph printed, a new point is supposed to appear. But we can see that the old ones are gone. That's because the function fit() reset each time it's called. Meanshift can't be used for sequential processing in Scikitlearn.
