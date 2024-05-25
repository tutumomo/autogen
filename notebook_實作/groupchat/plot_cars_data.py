# filename: plot_cars_data.py
import pandas as pd
import matplotlib.pyplot as plt

# Download the data
url = "https://raw.githubusercontent.com/uwdata/draco/master/data/cars.csv"
data = pd.read_csv(url)

# Select only the 'weight' and 'horsepower' columns
data_subset = data[['weight', 'horsepower']]

# Plot weight vs horsepower
plt.scatter(data_subset['weight'], data_subset['horsepower'])
plt.xlabel('Weight (lbs)')
plt.ylabel('Horsepower')
plt.title('Relationship between Weight and Horsepower')
plt.grid(True)
plt.savefig('cars_plot.png')
plt.show()