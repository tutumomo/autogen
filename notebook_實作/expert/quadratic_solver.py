# filename: quadratic_solver.py
import cmath

# Coefficients of the quadratic equation
a = 1
b = 3
c = 2

# Calculate the discriminant
discriminant = (b**2) - (4*a*c)

# Find two solutions
sol1 = (-b - cmath.sqrt(discriminant)) / (2*a)
sol2 = (-b + cmath.sqrt(discriminant)) / (2*a)

print("The solutions to the equation x^2 + 3x + 2 = 0 are:", sol1, "and", sol2)