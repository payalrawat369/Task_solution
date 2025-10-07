'''task1write code to create google form which has aldatatypes using input
 command- giving user input at runtime.
 using al5 methods: coma, + , % , .format , f string'''
name =input("Enter your name: ")
age = int(input("Enter your age: "))
cgpa= float(input("Enter your CGPA: "))
print("Using comma:", "Name:", name, "Age:", age, "CGPA:", cgpa)
print("Using +: " + "Name: " + name + " \n Age: " + str(age)+ " \n CGPA: " + str(cgpa))
print("Using %%: Name: %s \n Age: %d \n CGPA: %f " ,(name, age, cgpa))
print("Using .format(): Name: {} \n Age: {} \n CGPA: {:.2f}".format(name, age,cgpa))
print(f"Using f-string: Name: {name} \n Age: {age} \n CGPA: {cgpa:.2f}")

'''task2-functions manipulating lists only'''
def add_element(my_list, element):
    my_list.append(element)
    return my_list
# Function to remove an element
def remove_element(my_list, element):
    if element in my_list:
        my_list.remove(element)
    else:
        print(f"{element} not found in list")
        return my_list
# Function to sort the list
def sort_list(my_list):
    return sorted(my_list)
# Function to reverse the list
def reverse_list(my_list):
    return list(reversed(my_list))
# Function to find max and min
def min_max(my_list):
    return min(my_list), max(my_list)
# Function to search an element
def search_element(my_list, element):
    return element in my_list
numbers= [10, 5, 8, 20]

print("Original List:", numbers)
print("After Adding 15:", add_element(numbers, 15))
print("After Removing 8:", remove_element(numbers, 8))
print("Sorted List:", sort_list(numbers))
print("Reversed List:", reverse_list(numbers))
print("Min and Max:", min_max(numbers))
print("Search 20:", search_element(numbers, 20))
print("Search 50:", search_element(numbers, 50))

'''task3-builtin functions for dictionary'''
student = {
"name": "Payal",
"age": 20,
"cgpa": 8.6,
"course": "BTech"
 }
print("Dictionary:", student)
print("Keys:", student.keys())
print("Values:", student.values())
print("Items:", student.items())
print("Get name:", student.get("name"))
print("Get phone (with default):", student.get("phone", "Not Provided"))

 # pop()
removed_age = student.pop("age")
print("Removed age:", removed_age)
print("After pop:", student)
# popitem()
last_item = student.popitem()
print("Popped last item:", last_item)
print("After popitem:", student)
# update()
student.update({"semester": 5, "cgpa": 9.0})
print("After update:", student)
# len()
print("Number of items:", len(student))

