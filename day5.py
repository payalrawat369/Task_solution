#task1-do fibonacci & factorial of a number using recursive function
def factorial(n):
    if n == 0 or n == 1: 
        return 1
    else:
        return n * factorial(n - 1) 
num = int(input("Enter a number to find factorial: "))
print(f"Factorial of {num} is {factorial(num)}")

def fibonacci(n):
    if n == 0:
        return 0  
    elif n == 1:
        return 1 
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)  

num = int(input("Enter the position in Fibonacci series: "))
print(f"Fibonacci number at position {num} is {fibonacci(num)}")

#task2: write example code for modules OS, MATH, RANDOM
#OS module
import os

# 1. Get the current working directory
current_dir = os.getcwd()
print("Current Directory:", current_dir)

# 2. Create a new directory
new_dir = "TestFolder"
os.mkdir(new_dir)
print(f"Directory '{new_dir}' created.")

# 3. Change to the new directory
os.chdir(new_dir)
print("Changed working directory to:", os.getcwd())

# 4. List files and folders in current directory
files = os.listdir()
print("Contents of current directory:", files)

# 5. Change back to original directory
os.chdir(current_dir)
print("Back to original directory:", os.getcwd())

# 6. Remove the newly created directory
os.rmdir(new_dir)
print(f"Directory '{new_dir}' removed.")

#MATH module
#constants
import math

print("Pi:", math.pi)      # 3.14159...
print("Euler's number e:", math.e)  # 2.71828...

#trignometric 
# Example: sin(30 degrees)
angle = 30
radians = math.radians(angle)  # convert 30° to radians
print("sin(30°) =", math.sin(radians))
print("cos(30°) =", math.cos(radians))
print("tan(30°) =", math.tan(radians))

#logarithimic and power functions
print("sqrt(16):", math.sqrt(16))
print("2^3:", math.pow(2, 3))
print("ln(2):", math.log(2))
print("log10(100):", math.log10(100))
print("e^2:", math.exp(2))

#representation
print("ceil(4.2):", math.ceil(4.2))   
print("floor(4.8):", math.floor(4.8)) 

#RANDOM MODULE
import random

# 1. random
rand_float = random.random()
print("Random float between 0 and 1:", rand_float)

# 2. randint
rand_int = random.randint(10, 50)
print("Random integer between 10 and 50:", rand_int)

# 3. randrange
rand_range = random.randrange(0, 100, 5)  # multiples of 5 from 0 to 95
print("Random number from range 0-100 with step 5:", rand_range)

# 4. choice(sequence) 
colors = ['red', 'blue', 'green', 'yellow']
rand_choice = random.choice(colors)
print("Randomly chosen color:", rand_choice)

# 5. shuffle(list) 
numbers = [1, 2, 3, 4, 5]
random.shuffle(numbers)
print("Shuffled list:", numbers)

#task 3 :
# Function demonstrating *args (positional) and **kwargs (keyword) arguments
def demo_args(*args, **kwargs):
    print("Positional arguments (*args):", args)
    print("Keyword arguments (**kwargs):", kwargs)
    
    #for list
    squares = [x**2 for x in args]
    print("List comprehension (squares):", squares)
    
    #tuple
    cubes = tuple(x**3 for x in args)
    print("Tuple comprehension (cubes):", cubes)
    
    # Dictionary comprehension:
    doubled_values = {k: v*2 for k, v in kwargs.items()}
    print("Dictionary comprehension (doubled values):", doubled_values)
demo_args(1, 2, 3, 4, a=10, b=20, c=30)






    

