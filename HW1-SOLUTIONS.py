# PROBLEM1

#INTRODUCTION

#Say "Hello, World!" With Python
if __name__ == '__main__':
    my_string = "Hello, World!"
    print(my_string)

#Python If-Else

import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(input().strip())
    if n%2==1:
        print('Weird')
    else:
        if n>=2 and n<=5:
            print('Not Weird')
        elif n>=6 and n<=20:
            print('Weird')
        else:
            print('Not Weird')
#Arithmetic Operators

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a+b)
    print(a-b)
    print(a*b)

#Python: Division

if __name__ == '__main__':
    a = int(input())
    b = int(input())
    print(a//b)
    print(a/b)

#Loops

if __name__ == '__main__':
    n = int(input())
    i=0
    while i<n:
        print(i**2)
        i+=1

#Write a function

def is_leap(year):
    leap = False
    if year%4==0:
        if year%100==0:
            if year%400==0:
                leap=True
        else:
            leap=True
            
    
    return leap

#Print Function

if __name__ == '__main__':
    n = int(input())
    i=1
    x=''
    while i<=n:
        x+=str(i)
        i+=1
print(x)

#DATA TYPES

#List Comprehensions

if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    a=[]
    for i in range(0,x+1):
        for j in range(0,y+1):
            for k in range(0,z+1):
                a.append([i,j,k])
   
        
                
    
    newlist = [q for q in a if (q[0]+q[1]+q[2])!=n]
    print(newlist)

#Find the Runner-Up Score!

if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    b=list(arr)
    c=[x for x in b if x!=max(b)]
    c.sort()
    print(c[-1])

#Nested Lists

if __name__ == '__main__':
    a=[]
    for _ in range(int(input())):
        name = input()
        score = float(input())
        a.append([name,score])
    c=[]
    for i in range(0,len(a)):
        c.append(a[i][1])
    b=min(c)
    newlist=[x for x in a if x[1]!=b]
    d=[]
    for i in range(0,len(newlist)):
        d.append(newlist[i][1])
    z=min(d)
    newlist2=[x for x in newlist if x[1]==z]
    q=[]
    for x in newlist2:
        q.append(x[0])
    q.sort()
    for x in q:
        print(x)

#Finding the percentage

if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    a=student_marks[query_name]
    print("{:.2f}".format((sum(a)/len(a))))

#Tuples

if __name__ == '__main__':
    n = int(input())
    integer_list = map(int, input().split())
    print(hash(tuple(integer_list)))

#STRINGS

#sWAP cASE

def swap_case(s):
    a=''
    for x in range(len(s)):
        if s[x].islower():
            a+=s[x].upper()
            
        else:
            a+=s[x].lower()
    return a

#String Split and Join

def split_and_join(line):
    a=line.split(' ')
    a="-".join(a)
    return a

#What's Your Name?

def print_full_name(first, last):
    print('Hello '+ first +' ' +last+'! You just delved into python.')

#Mutations

def mutate_string(string, position, character):
    string = string[:position] + character + string[position+1:]
    return string

#Find a string

def count_substring(string, sub_string):
    num=0
    for x in range(len(string)-len(sub_string)+1):
        if string[x:len(sub_string)+x]==sub_string:
            num+=1
            
    return num
    
#String Validators

if __name__ == '__main__':
    s = input()
    for x in range(len(s)):
        if s[x].isalnum():
            print(True)
            break
    else: 
        print('False')
    for x in range(len(s)):
        if s[x].isalpha():
            print(True)
            break
    else: 
        print('False')
    
    for x in range(len(s)):
        if s[x].isdigit():
            print(True)
            break
    else: 
        print('False')
    for x in range(len(s)):
        if s[x].islower():
            print(True)
            break
    else: 
        print('False')
    
    for x in range(len(s)):
        if s[x].isupper():
            print(True)
            break
    else: 
        print('False') 

#Text Alignment

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

#Text Wrap

def wrap(string, max_width):
    for i in range(0,len(string),max_width):
        result = string[i:i+max_width]
        if len(result) == max_width:
            print(result)
        else:
            return(result)

#Capitalize!

def solve(s):
    for x in s.split():
        s = s.replace(x, x.capitalize())
    return s


#SETS

#Introduction to Sets

def average(array):
    return "{:.3f}".format(sum(set(array))/len(set(array)))

#No Idea!

n,m=map(int,input().split())
l=list(map(int,input().split()))
a=set(map(int,input().split()))
b=set(map(int,input().split()))
happiness=0
for x in l:
    if x in a:
        happiness+=1
    elif x in b:
        happiness-=1
print(happiness)

#Symmetric Difference

n1=int(input())
s1=set(map(int,input().split()))
n2=int(input())
s2=set(map(int,input().split()))
s3=s1.symmetric_difference(s2)
for x in sorted(s3):
    print(x)

#Set.add()

n=int(input())
s=set()
for x in range(n):
    s.add(input())
print(len(s))

#Set.union() Operation

nenglish=int(input())
english=set(map(int,input().split()))
nfrench=int(input())
french=set(map(int,input().split()))
print(len((english.union(french))))

#Set .intersection() Operation

nenglish=int(input())
english=set(map(int,input().split()))
nfrench=int(input())
french=set(map(int,input().split()))
print(len((english.intersection(french))))

#Set .difference() Operation

nenglish=int(input())
english=set(map(int,input().split()))
nfrench=int(input())
french=set(map(int,input().split()))
print(len((english.difference(french))))

#Set .symmetric_difference() Operation

nenglish=int(input())
english=set(map(int,input().split()))
nfrench=int(input())
french=set(map(int,input().split()))
print(len((english.symmetric_difference(french))))

#The Captain's Room

n = input()
roomlist = input().split()
roomset = set(roomlist)
for ele in list(roomset):
    roomlist.remove(ele)
    
x = roomset.difference(set(roomlist)).pop()
print(x)

#Check Subset

t=int(input())
for x in range(t):
    na=int(input())
    a=set(map(int,input().split()))
    nb=int(input())
    b=set(map(int,input().split()))
    if na>nb:
        print(False)
    elif len(a.union(b))==len(b):
        print(True)
    else: 
        print(False)

#Check Strict Superset

a=set(map(int,input().split()))
n=int(input())
q=True
for x in range(n):
    b=set(map(int,input().split()))
    if a.__eq__(b):
        q=False
        break
    elif len(a)<=len(b):
        q=False
        break
    for x in b:
        if x not in a:
            q=False
            break
print(q)

#COLLECTIONS

#collections.Counter()

from collections import Counter
numshoes=int(input())
shoes=Counter(map(int,input().split()))
numcust=int(input())
income=0
for i in range(numcust):
    size, price = map(int, input().split())
    if shoes[size]: 
        income += price
        shoes[size] -= 1
print(income)

#Collections.namedtuple()

from collections import namedtuple
n = int(input())
w = input().split()
x = 0
for _ in range(n):
    students = namedtuple('student', w)
    voti, classe, nome, Id = input().split()
    student = students(voti, classe, nome, Id)
    x += int(student.MARKS)
print('{:.2f}'.format(x / n))

#Collections.deque()

from collections import deque
D = deque()
for _ in range(int(input())):
    oper, val, *args = input().split() + ['']
    eval(f'D.{oper} ({val})')
print(*D)

#Collections.OrderedDict()

from collections import OrderedDict
order = OrderedDict()
for _ in range(int(input())):
    item, space, price = input().rpartition(' ')
    order[item] = order.get(item, 0) + int(price)
for item, price in order.items():
    print(item, price)

# DATE AND TIME 

#Calendar Module

import datetime
import calendar
m, d, y = map(int, input().split())
input_date = datetime.date(y, m, d)
print(calendar.day_name[input_date.weekday()].upper())

#Time Delta

import math
import os
import random
import re
import sys

from datetime import datetime
def time_delta(t1, t2):
    time_format = '%a %d %b %Y %H:%M:%S %z'
    t1 = datetime.strptime(t1, time_format)
    t2 = datetime.strptime(t2, time_format)
    return str(int(abs((t1-t2).total_seconds()))) 

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()


#EXCEPTIONS

#Exceptions

n=int(input())
ins=[]
for i in range(n):
    ins.append(input().split())
for i in range(n):
    try:
        print(int(ins[i][0])//int(ins[i][1]))
    except ZeroDivisionError as e:
        print ("Error Code:",e)
    except ValueError as e:
        print ("Error Code:",e)

# BUILT-INS

#Zipped!

numstud,numsubject=map(int,input().split())
lis=list()
for x in range(numsubject):
    num=map(float,input().split())
    lis.append(num)
for i in zip(*lis):
    print(sum(i)/len(i))

#ginortS

print(*sorted(input(), key=lambda c: (c.isdigit() - c.islower(), c in '02468', c)), sep='')

# PYTHON FUNCTIONALS

#Map and Lambda Function

cube = lambda x: x**3

def fibonacci(n):
    lis=[0,1]
    if n==0:
        return []
    if n==1:
        return [0]
    if n==2:
        return lis
    else:
        for x in range(2,n):
            lis.append(lis[x-1]+lis[x-2])
        
    return lis

# REGEX AND PARSING CHALLENGES

#Detect Floating Point Number

from re import match, compile
pattern = compile('^[-+]?[0-9]*\.[0-9]+$')
for _ in range(int(input())):
    print(bool(pattern.match(input())))

#Re.split()

regex_pattern = r"[.,]+"

import re
print("\n".join(re.split(regex_pattern, input())))

#Group(), Groups() & Groupdict()

import re
m = re.search(r'([a-zA-Z0-9])\1', input().strip())
print(m.group(1) if m else -1)

#Re.findall() & Re.finditer()

import re
vocali = 'aeiou'
consonanti = 'qwrtypsdfghjklzxcvbnm'
match = re.findall(r'(?<=[' + consonanti + '])([' + vocali + ']{2,})(?=[' + consonanti + '])', input(), flags=re.I)
print('\n'.join(match or ['-1']))

#Re.start() & Re.end()

import re
s = input()
ss = input()
pattern = re.compile(ss)
match = pattern.search(s)
if not match: print('(-1, -1)')
while match:
    print('({0}, {1})'.format(match.start(), match.end() - 1))
    match = pattern.search(s, match.start() + 1)

#Validating Roman Numerals

regex_pattern = r'M{0,3}(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[VX]|V?I{0,3})$'# Do not delete 'r'.

import re
print(str(bool(re.match(regex_pattern, input()))))

#Validating phone numbers

import re
[print('YES' if re.match(r'[789]\d{9}$', input()) else 'NO') for _ in range(int(input()))]

#Validating and Parsing Email Addresses

import re
pattern = r'^<[A-Za-z](\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>$'
for _ in range(int(input())):
    nome, email = input().split(' ')
    if re.match(pattern, email):
        print(nome, email)

#Hex Color Code

import re
for _ in range(int(input())):
    matches = re.findall(r':?.(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3})', input())
    if matches:
        print(*matches, sep='\n')

#Validating UID


import re
for _ in range(int(input())):
    s = ''.join(sorted(input()))
    try:
        assert re.search(r'[A-Z]{2}', s)
        assert re.search(r'\d\d\d', s)
        assert not re.search(r'[^a-zA-Z0-9]', s)
        assert not re.search(r'(.)\1', s)
        assert len(s) == 10
    except:
        print('Invalid')
    else:
        print('Valid')

# CLOSURES AND DECORATIONS

#Standardize Mobile Number Using Decorators

def wrapper(f):
    def fun(l):
        f(['+91 ' + x[-10:-5] + ' ' + x[-5:] for x in l])
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l) 

#Decorators 2 - Name Directory

import operator

def person_lister(f):
    def inner(people):
        return [f(a) for a in sorted(people, key = lambda x: (int(x[2])))]
    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')


#NUMPY

 #Arrays

import numpy

def arrays(arr):
    a=numpy.array(arr,float)
    return (a[::-1])

arr = input().strip().split(' ')
result = arrays(arr)
print(result)

#Shape and Reshape


import numpy
l=list(map(int,input().split()))
t=numpy.array(l)
print(numpy.reshape(t,(3,3)))

#Transpose and Flatten

import numpy
n,m=map(int,input().split())
lis=[]
for x in range(n):
    lis.append(list(map(int,input().split())))
arr=numpy.array(lis)
print(arr.transpose())
print(arr.flatten())

#CONCATENATE

import numpy
n,m,p=map(int,input().split())
l1=[]
l2=[]
for x in range(n):
    l1.append(list(map(int,input().split())))
for x in range(m):
    l2.append(list(map(int,input().split())))
arr1=numpy.array(l1)
arr2=numpy.array(l2)
print(numpy.concatenate((arr1, arr2), axis = 0)) 

#ZEROS AND ONES

import numpy as np
shape=tuple(map(int,input().split()))
print(np.zeros(shape,int))
print(np.ones(shape,int))

#EYE AND IDENTITY

import numpy
numpy.set_printoptions(legacy='1.13')
n,m=map(int,input().split())
print(numpy.eye(n,m))

#Array Mathematcis

import numpy
n,m=map(int,input().split())
l1=[]
l2=[]
for x in range(n):
    l1.append(list(map(int,input().split())))
for x in range(n):
    l2.append(list(map(int,input().split())))
arr1=numpy.array(l1)
arr2=numpy.array(l2)
print(arr1+arr2)
print(arr1-arr2)
print(arr1*arr2)
print(arr1//arr2)
print(arr1%arr2)
print(arr1**arr2)

# Floor, Ceil and Rint

import numpy
numpy.set_printoptions(legacy='1.13')
a=numpy.array(list(map(float,input().split())))
print(numpy.floor(a),numpy.ceil(a),numpy.rint(a),sep='\n')

# Sum and Prod 

import numpy
n,m=map(int,input().split())
l=[]
for x in range(n):
    l.append(list(map(int,input().split())))
a=numpy.array(l)
sum0=numpy.sum(a,axis=0)
print(numpy.prod(sum0))

# Min and Max

import numpy
n,m=map(int,input().split())
l=[]
for _ in range(n):
    l.append(list(map(int,input().split())))
a=numpy.array(l)
print(numpy.max(numpy.min(a,axis=1)))

#Mean, Var, and Std

import numpy
n,m=map(int,input().split())
l=[]
for x in range(n):
    l.append(list(map(int,input().split())))
a=numpy.array(l)
print(numpy.mean(a,axis=1),numpy.var(a,axis=0),round(numpy.std(a),11),sep='\n')

#Dot and Cross

import numpy
n=int(input())
l1=[]
l2=[]
for x in range(n):
    l1.append(list(map(int,input().split())))
a1=numpy.array(l1)
for x in range(n):
    l2.append(list(map(int,input().split())))
a2=numpy.array(l2)
print(numpy.dot(a1,a2))

# Inner and Outer

import numpy
a=numpy.array(list(map(int,input().split())))
b=numpy.array(list(map(int,input().split())))
print(numpy.inner(a,b),numpy.outer(a,b),sep='\n')

# Polynomials
import numpy
l=list(map(float,input().split()))
x=float(input())
print(numpy.polyval(l, x))

# Linear Algebra

import numpy
n=int(input())
l=[]

for _ in range(n):
    l.append(list(map(float,input().split())))
a=numpy.array(l)
print(round(numpy.linalg.det(a),2))

# PROBLEM 2

#Birthday Cake Candles

import math
import os
import random
import re
import sys

def birthdayCakeCandles(candles):
    m=max(candles)
    l=[x for x in candles if x==m]
    return len(l)
        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()


#Number Line Jumps Kangaroo


import math
import os
import random
import re
import sys


def kangaroo(x1, v1, x2, v2):
    if x2==x1 and v2==v1:
        return 'YES'
    elif x2>=x1 and v2>=v1:
        return 'NO'
    elif x1>=x2 and v1>=v2:
        return 'NO'
    else:
        if (x2-x1)%(v1-v2)==0:
            return 'YES'
        else:
            return 'NO'
        
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()

#Viral Advertising


import math
import os
import random
import re
import sys

def viralAdvertising(n):
    s=5
    likes=[]
    for x in range(n):
        likes.append(s//2)
        s=3*likes[x]
    return sum(likes)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()

#Insertion Sort - Part 1

import math
import os
import random
import re
import sys

def insertionSort1(n, arr):
    x=arr[-1]
    indice=n-2

    while (x < arr[indice]) and (indice >= 0):
        arr[indice+1]=arr[indice]
        print(*arr)
        indice-=1

    arr[indice+1]=x
    print(*arr)

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)

#Insertion Sort - Part 2


import math
import os
import random
import re
import sys


def insertionSort2(n, arr):
    # Write your code here
    for i in range(1, len(arr)):
        x = arr[i]
        k = i-1
        while arr[k]> x and k>=0:
            arr[k+1] = arr[k]
            k-=1
        arr[k+1] = x   
        print(*arr)
if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)
