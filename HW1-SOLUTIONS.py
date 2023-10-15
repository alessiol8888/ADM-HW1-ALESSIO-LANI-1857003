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

#
  
