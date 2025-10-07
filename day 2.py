'''task1:write code for
  1.Numeric:integer(positive,negative,binary,octal,hexa)
  2.float
  3.complex
  4.Boolean'''

#1.Numeric:integer(positive,negative,binary,octal,hexa)
pos_int=62;
neg_int=-45;
bol_int=0b1010;
oct_int=0o52;
hex_int=0x2A;

print("positive integer",pos_int)
print("negative integer",neg_int)
print("boolean integer",bol_int)
print("octal integer",oct_int)
print("hexadecimal integer",hex_int)

#2.float
float_num=45.236
print("float number",float_num)

#complex
comp_num=5+5j;
print("complex numer:",comp_num);

#boolean
bool_true = True
bool_false = False
print("Boolean True:", bool_true)
print("Boolean False:", bool_false)


'''task2:
# write code for all arithmatic operators'''

# Taking input from user
num1 = float(input("Enter first number: "))
num2 = float(input("Enter second number: "))

# Addition
add = num1 + num2
print("Addition (+):", add)

# Subtraction
sub = num1 - num2
print("Subtraction (-):", sub)

# Multiplication
mul = num1 * num2
print("Multiplication (*):", mul)

# Division
div = num1 / num2
print("Division (/):", div)

# Floor Division
floor_div = num1 // num2
print("Floor Division (//):", floor_div)

# Modulus (Remainder)
mod = num1 % num2
print("Modulus (%):", mod)

# Exponentiation
exp = num1 ** num2
print("Exponentiation (**):", exp)


'''task3:write string
  1.concatenation
  2.repetition
  3.slice
  4.range slice
  5.In Membership[in]
  6.Not in membership[not in]'''

# Task 3: String Operations

# Taking string input from user
str1 = input("Enter first string: ")
str2 = input("Enter second string: ")

# 1. Concatenation (+)
concat = str1 + str2
print("Concatenation (+):", concat)

# 2. Repetition (*)
repeat = str1 * 3
print("Repetition (*):", repeat)

# 3. Slice [start:end] (extract substring)
slice_str = str1[1:5]  # characters from index 1 to 4
print("Slice [1:5]:", slice_str)

# 4. Range Slice with step [start:end:step]
range_slice = str1[0:10:2]  # characters from index 0 to 9, step 2
print("Range Slice [0:10:2]:", range_slice)
 
# 5. Membership [in]
check_in = "a" in str1
print("'a' in string:", check_in)

# 6. Not in membership [not in]
check_not_in = "z" not in str1
print("'z' not in string:", check_not_in)
