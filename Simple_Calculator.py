def add(a, b):
    print(f"Adding {a} and {b}")
    return a + b
def subtract(a, b):
    print(f"Subtracting {a} and {b}")
    return a - b
def multiply(a, b): 
    print(f"Multiplying {a} and {b}")
    return a * b
def divide(a, b): 
    print(f"Dividing {a} and {b}...")
    return a / b

print("Let's do some math with just functions")

#taking user input for the numbers
number1 = int(input("What is the first number? \n >"))
number2 = int(input("What is the second number? \n >"))
add_result = add(number1, number2)
subtract_result = subtract(number1, number2)
multiply_result = multiply(number1, number2)
divide_result = divide(number1, number2)

print(f"N1: {number1}, N2: {number2}, \n\t Addition: {add_result}, \n\t Subtraction: {subtract_result}, \n\t Multiplication: {multiply_result}, \n\t Division: {divide_result}.")
