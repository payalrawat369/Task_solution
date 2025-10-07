'''task1-Write code for calculator with all 7 operators + - * / % // ** using if elif else'''

num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))

operator = input("\nEnter your operator: ")

if operator == '+':
    print("Result:", num1 + num2)
elif operator == '-':
    print("Result:", num1 - num2)
elif operator == '*':
    print("Result:", num1 * num2)
elif operator == '/':
    if num2 != 0:
        print("Result:", num1 / num2)
    else:
        print("Error: Division by zero is not allowed.")
elif operator == '%':
    if num2 != 0:
        print("Result:", num1 % num2)
    else:
        print("Error: Modulus by zero is not allowed.")
elif operator == '//':
    if num2 != 0:
        print("Result:", num1 // num2)
    else:
        print("Error: Floor division by zero is not allowed.")
elif operator == '**':
    print("Result:", num1 ** num2)
else:
    print("Invalid operator! Please select a valid one.")

#task2-for loop using tuple , dictionary , set
# --- For loop with Tuple ---
print("Tuple Example:")
my_tuple = (10, 20, 30, 40, 50)
for item in my_tuple:
    print(item)

# --- For loop with Dictionary ---
print("\nDictionary Example:")
my_dict = {"name": "Payal", "age": 19, "course": "BTech"}
for key, value in my_dict.items():
    print(key, ":", value)

# --- For loop with Set ---
print("\nSet Example:")
my_set = {100, 200, 300, 400}
for element in my_set:
    print(element)

#task3-design calculator using structural pattern matching

a = float(input("Enter first number: "))
b = float(input("Enter second number: "))
op = input("Enter operator(+, -, *, /, %, **): ")

match op:
    case '+':
        result = a + b
    case '-':
        result = a - b
    case '*':
        result = a * b
    case '/':
        result = "Error: Division by zero" if b == 0 else a / b
    case '%':
        result = a % b
    case '**':
        result = a ** b
    case _:
        result = "Invalid operator"

print("Result:", result)

#task4-write all file concept commands
import os

#Check if File Exists
print("\n Checking if file exists:")
if os.path.exists("lecture.txt"):
    print("lecture.txt exists")
else:print("not exists")

# Create and Write 
print("\nCreating and Writing to lecture.txt")
f = open("lecture.txt", "w")
f.write("This is the first line.\n")
f.write("This is the second line.\n")
f.write("This is the third line.\n")
f.close()

#Read Entire File
print("\nReading entire file:")
f = open("lecture.txt", "r")
print(f.read())
f.close()

#  Read First 10 Characters
print("\nReading first 10 characters:")
f = open("lecture.txt", "r")
print(f.read(10))
f.close()

#Read Line by Line
print("\nReading line by line:")
f = open("lecture.txt", "r")
print(f.readline())  
print(f.readline()) 
f.close()

#Read All Lines into a List 
print("\nReading all lines into list:")
f = open("lecture.txt", "r")
print(f.readlines())
f.close()

# Append to File 
print("\nAppending new line:")
f = open("lecture.txt", "a")
f.write("This is an appended line.\n")
f.close()

# Read After Append 
print("\nReading after append:")
with open("lecture.txt", "r") as f:
    print(f.read())

# File Pointer (seek & tell)
print("\nUsing seek() and tell():")
f = open("lecture.txt", "r")
print("First 5 chars:", f.read(5))
print("Pointer position:", f.tell())
f.seek(0)
print("After seek -> first 5 chars again:", f.read(5))
f.close()






