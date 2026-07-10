# simple_error.py

name = input("Enter your name: ")
age = input("Enter your age: ")

print("Hello " + name)

# Error 1: age is a string, but adding an integer
next_year = age + 1

print("Next year you will be", next_year)

numbers = [10, 20, 30]

# Error 2: List index out of range
print("Fourth number:", numbers[3])

# Error 3: Variable name typo
total = 100
print("Total is", totall)

# Error 4: Division by zero
result = 100 / 0

print("Result:", result)
