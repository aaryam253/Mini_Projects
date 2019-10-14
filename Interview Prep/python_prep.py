import pandas as pd
'''
# Fizz Buzz
for num in range(1, 101):
	if num % 5 == 0 and num % 3 == 0:
		print("FizzBuzz")
	elif num % 3 == 0:
		print ("Fizz")
	elif num % 5 == 0:
		print ("Buzz")
	else: 
		print num

# Fibonacci Seq.
a, b = 0, 1
for i in xrange(0, 10):
	print a
	a, b = b, a + b

# Lists
my_list = [1,2,3,4]
for i in my_list:
	print i

# tuples
my_tup = (1,2,3,4)

# Dictionaries
my_dict = {'monday':10, 'tuesday': 20}

# Set
my_set = {1,2,3,4,4,5,6,6}

# list comprehension
squares = [x*x for x in my_list]

# generators
def fib(num):
	a,b = 0,1
	for i in xrange(0, num):
		yield "{}: {}".format(i+1, a)
		a, b = b, a+b

for item in fib(10):
	print(item)

class Person(self):
	def __init__(self, name):
		self.name = name

	def reveal_iden(self):
		print ("My name is {}".format(self.name))

class SuperHero(Person):
	def __init__(self, name, hero_name):
		super().__init__(name)
		self.hero_name = hero_name

	def reveal_iden(self):
		super().reveal_iden()
		print("I am a {}".format(self.hero_name))
'''

# Question Types
# NumPy --> linear algebra library in Python
import numpy as np
a = np.array([1,2,3])
a2 = np.array([1,2,3])
b = np.array([[1,2,3],[4,5,6]])
c = np.zeros((5,5))
print(np.sum((a,a2), ax))

# Dataframe
l1 = [1,2,3,4,5]
data1 = pd.DataFrame(l1)
print(data1)






