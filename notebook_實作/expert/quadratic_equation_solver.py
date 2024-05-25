# filename: quadratic_equation_solver.py
import math

# coefficients
a = 1
b = 3
c = 2

# calculate the discriminant
discriminant = b**2 - 4*a*c

# calculate the solutions for x
if discriminant > 0:
    x1 = (-b + math.sqrt(discriminant)) / (2*a)
    x2 = (-b - math.sqrt(discriminant)) / (2*a)
    print("The solutions are:", x1, "and", x2)
elif discriminant == 0:
    x = -b / (2*a)
    print("The only solution is:", x)
else:
    print("The equation has no real solutions")