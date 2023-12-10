import numpy as np
import matplotlib.pyplot as plt

# Load histogram data from file
histogram_data = np.loadtxt('histogram.txt')

# Create bins (assuming equally spaced bins)
bins = np.arange(len(histogram_data))

# Plot the histogram
plt.bar(bins, histogram_data, width=1.0, align='edge')

# Add labels and title
plt.xlabel('Bin Index')
plt.ylabel('Count')
plt.title('Histogram Plot')

# Show the plot
# plt.show()

# Save the figure to a file
plt.savefig('histogram.png')