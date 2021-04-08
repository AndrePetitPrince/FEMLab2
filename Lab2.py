#http://davinci.mimuw.edu.pl/book/export/html/51


#print ("Lorem ipsum ...")

"""
tekst = "Lorem ipsum dolor sit amet, ..."
print (tekst)
"""

"""
p = 8.5 * (67*0.1 + 124*0.15 + 11.7*0.2)
a = 124*0.17
pr = 50
k = p + a + pr
print (k)
print (k / 2)
print (k / 3)
print (k / 4)
print (k / 5)
"""

"""
zmienna = "Napis" 
print (zmienna) 
zmienna = 8.5 
print (zmienna)
"""

"""
d=[10, 8, 10, 12, 6, 8, 7, 12, 10, 16, 16, 9, 14, 9, 11, 17, 18, 9, 5, 17, 11, 17, 7, 7, 12, 9, 5, 18, 6, 7, 9, 9, 6, 8, 8, 11, 13, 16, 8, 8, 12, 5, 18, 15, 17, 18, 7, 8, 13, 5, 12, 11, 11, 12, 5, 17, 7, 15, 10, 14, 18, 5, 8, 9, 10, 14, 15, 13, 16, 14, 17, 16, 10, 7, 14, 15, 17, 11, 10, 18, 18, 9, 12, 18, 12, 13, 7, 10, 16, 12, 16, 8, 11, 15, 8, 7, 7, 10, 13, 13]
print (d[16])
print (d[9:15])
print (d[:5])
print (d[-1])

print (d[11])
d[11] = d[11] - 0.1
print (d[11])

d.append(14)
print (d)
for i in range(len(d)):
    d[i] = d[i]*0.1
print (d)
"""
"""
for element in d:
    print (element)
"""
"""
for element in [1,2,3,4,5]:
  print (element)
print ("zrobimy sobie odstep")
print (element)
"""

"""
dictionary = {'plytka1':101, 'plytka2':201, 'plytka3': 302, 'plytka4':102}
print (dictionary['plytka1'])
print (dictionary)
print (dictionary.keys())
print (dictionary.values())

dictionary['plytka5']=0
print (dictionary)

del (dictionary['plytka5'])
print (dictionary)
del (dictionary)
print (dictionary)
"""

"""
print (len([851, 1, 58]))
"""

"""
zmienna = 8
if zmienna > 5:
    print ("Zmienna jest wieksza niz 5")
else:
    print ("Zmienna nie jest rowna 15")
"""

"""
zmienna = 8
if zmienna == 1:
    print ("Wartosc zmiennej to jeden")
else:
    if zmienna == 2:
        print ("Wartosc zmiennej to dwa")
    else:
        print ("Wartosc zmiennej nie jest ani jeden ani dwa")
"""

"""
zmienna = 8
if zmienna == 5:
    print ("Zmienna jest rowna 5")
elif zmienna == 8:
    print ("Zmienna jest rowna 8")
elif zmienna > 5:
    print ("Zmienna jest rowna 5 ale nie jest rowna 8")
else:
    print ("Zmienna jest mniejsza od 5")
"""

"""
x=0
while x<5:
    print ("zmienna x wynosi ", x)
    x+=1
"""

"""
x=0
while True:
    print ("To wyswietli sie zawsze gdy warunek jest spelniony")
    if x==5:
        print ("zmienna x wynosi ", x)
        break
    else:
        print (x)
    x+=1 
"""

"""
x=[1,2,3,4,5]
while x:
    y=x.pop()
    print ("ostatnia wartość z listy x to ", y)
else:
    print ('koniec')
"""

"""
x = 15
wynik = (2 * (x**3))/8.51
print (wynik)
"""

"""
def f(x):
    return (2 * (x**3))/8.51
print (f(5))
"""

"""
def funkcja():
    print ("wykonalo sie")
funkcja()
"""

"""
def f(x=0):
    return (2 * (x**3))/8.51
 
print (f())
print (f(5))
"""



