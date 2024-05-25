# filename: plot_cars.py

import pandas as pd
import matplotlib.pyplot as plt

# Download the dataset
url = "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv"
df = pd.read_csv(url, encoding='utf-8')

# Print the fields in the dataset
print(df.columns)

# Plot the relationship between weight and horsepower
plt.scatter(df['Weight'], df['Horsepower'])
plt.xlabel('Weight')
plt.ylabel('Horsepower')
plt.title('Relationship between Weight and Horsepower')

# Improve aesthetics
plt.scatter(df['Weight'], df['Horsepower'], c='blue', alpha=0.5, s=50)
plt.legend(['Data Points'])
plt.colorbar()

# Show the plot
plt.show()

# Save the plot to a file
plt.savefig("weight_horsepower_plot.png")