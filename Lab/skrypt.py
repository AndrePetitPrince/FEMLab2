import numpy as np

#4 NumPy
#4.1 Tworzenie tablic
#4.1.1 Wymienienie elementów
arr = np.array([1, 2, 3, 4, 5])
print(arr)
print("\n")

A = np.array([[1, 2, 3], [7, 8, 9]])
print(A)
print("\n")

A = np.array([[1, 2, 3],
    [7, 8, 9]])
print(A)
print("\n")

#4.1.2 Generowanie elementów
v = np.arange(1,7)
print(v,"\n")
v = np.arange(-2,7)
print(v,"\n")
v = np.arange(1,10,3)
print(v,"\n")
v = np.arange(1,10.1,3)
print(v,"\n")
v = np.arange(1,11,3)
print(v,"\n")
v = np.arange(1,2,0.1)
print(v,"\n")
print("\n")

v = np.linspace(1,3,4)
print(v)
v = np.linspace(1,10,4)
print(v)
print("\n")

X = np.ones((2,3))
Y = np.zeros((2,3,4))
Z = np.eye(2) # np.eye(2,2) # np.eye(2,3)
Q = np.random.rand(2,5) # np.round(10*np.random.rand((3,3)))
print(X,"\n\n",Y,"\n\n",Z,"\n\n",Q)
print("\n")

#4.1.3 Budowanie z innych tablic (sklejanie)
U = np.block([[A],[X]])
print(U)
print("\n")

#4.1.4 Mieszanie powyzszych sposobów
V = np.block([[
np.block([
np.block([[np.linspace(1,3,3)],
[np.zeros((2,3))]]) ,
np.ones((3,1))])
],
[np.array([100, 3, 1/2, 0.333])]] )
print(V)
print("\n")

#4.2 Odwo³ywanie sie do elementów tablicy
print( V[0,2] )
print( V[3,0] )
print( V[3,3] )
print( V[-1,-1] )
print( V[-4,-3] )
print( V[3,:] )
print( V[:,2] )
print( V[3,0:3] )
print( V[np.ix_([0,2,3],[0,-1])] )
print( V[3] )
print("\n")

#4.3 Usuwanie fragmentów macierzy i tablic
Q = np.delete(V, 2, 0)
print(Q)
Q = np.delete(V, 2, 1)
print(Q)
v = np.arange(1,7)
print( np.delete(v, 3, 0) )
print("\n")

#4.4 Sprawdzanie rozmiarów tablic
print(np.size(v))
print(np.shape(v))
print(np.size(V))
print(np.shape(V))
print("\n")

#4.5 Operacje na macierzach
#4.5.1
A = np.array([[1, 0, 0],
    [2, 3, -1],
    [0, 7, 2]] )
B = np.array([[1, 2, 3],
    [-1, 5, 2],
    [2, 2, 2]] )
print( A+B )
print( A-B )
print( A+2 )
print( 2*A )
print("\n")

#4.5.2 Mnozenie macierzowe
MM1 = A@B
print(MM1)
MM2 = B@A
print(MM2)
print("\n")

#4.5.3 Mnozenie tablicowe
MT1 = A*B
print(MT1)
MT2 = B*A
print(MT2)
print("\n")

#4.5.4 Dzielenie tablicowe
DT1 = A/B
print(DT1)
print("\n")

#4.5.5 Dzielenie macierzowe - rozwiazywanie URL
C = np.linalg.solve(A,MM1)
print(C) # porownaj z macierza B
print(B)
print("\n")

x = np.ones((3,1))
b = A@x
y = np.linalg.solve(A,b)
print(y)
print("\n")

#4.5.6 Potegowanie
PM = np.linalg.matrix_power(A,2) # por. A@A
print(PM)
print(A@A)
print("\n")

PT = A**2 # por. A*A
print(PT)
print(A*A)
print("\n")

#4.5.7 Transpozycja
print(A.T) # transpozycja
print(A.transpose())
print("\n")

print(A.conj().T) # hermitowskie sprzezenie macierzy (dla m. zespolonych)
print(A.conj().transpose())
print("\n")

#4.6 Operacje porównan i funkcje logiczne
print(A)
print(B)
print(A == B)
print("\n")

print(A)
print(B)
print(A != B)
print("\n")

print(A)
print(2 < A)
print("\n")

print(A)
print(B)
print(A > B)
print("\n")

print(A)
print(B)
print(A < B)
print("\n")

print(A)
print(B)
print(A >= B)
print("\n")

print(A)
print(B)
print(A <= B)
print("\n")

print(A)
print(np.logical_not(A))
print("\n")

print(A)
print(B)
print(np.logical_and(A, B))
print("\n")

print(A)
print(B)
print(np.logical_or(A, B))
print("\n")

print(A)
print(B)
print(np.logical_xor(A, B))
print("\n")

print(A)
print( np.all(A) )
print("\n")

print(A)
print( np.any(A) )
print("\n")

print(v)
print( v > 4 )
print("\n")

print(v)
print( np.logical_or(v>4, v<2))
print("\n")

print(v)
print(np.nonzero(v>4))
print("\n")

print(v)
print( v[np.nonzero(v>4) ] )
print("\n")

#4.7 Inne
print(A)
print(np.max(A))
print(np.min(A))
print(np.max(A,0))
print(np.max(A,1))
print( A.flatten() )
print( A.flatten('F'))
print("\n")

#5 Matplotlib
#5.1 Wykresy funkcji
import matplotlib.pyplot as plt
x = [1,2,3]
y = [4,6,5]
plt.figure(1)
plt.plot(x,y)

#5.1.1 Rysujemy wykres funkcji sinus
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0.0, 2.0, 0.01)
y = np.sin(2.0*np.pi*x)
plt.figure(2)
plt.plot(x,y)

#5.1.2 Ulepszamy wykres
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0.0, 2.0, 0.01)
y = np.sin(2.0*np.pi*x)
plt.figure(3)
plt.plot(x,y,'r:',linewidth=6)
plt.xlabel('Czas')
plt.ylabel('Pozycja')
plt.title('Nasz pierwszy wykres')
plt.grid(True)

#5.1.3 Kilka wykresów we wspólnych osiach - Pierwsza wersja
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0.0, 2.0, 0.01)
y1 = np.sin(2.0*np.pi*x)
y2 = np.cos(2.0*np.pi*x)
plt.figure(4)
plt.plot(x,y1,'r:',x,y2,'g')
plt.legend(('dane y1','dane y2'))
plt.xlabel('Czas')
plt.ylabel('Pozycja')
plt.title('Wykres ')
plt.grid(True)

#5.1.4 Kilka wykresów we wspólnych osiach - Druga wersja
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0.0, 2.0, 0.01)
y1 = np.sin(2.0*np.pi*x)
y2 = np.cos(2.0*np.pi*x)
y = y1*y2
l1, = plt.plot(x,y,'b')
l2,l3 = plt.plot(x,y1,'r:',x,y2,'g')
plt.legend((l2,l3,l1),('dane y1','dane y2','y1*y2'))
plt.xlabel('Czas')
plt.ylabel('Pozycja')
plt.title('Wykres ')
plt.grid(True)

#plt.show()


#6 Cwiczenia
#Zadanie 3
A = np.block([[
    np.block([[
        np.block([[
            np.linspace(1,5,5,dtype=int)],
                  [np.linspace(5,1,5,dtype=int)]])],
              [np.block([[
                  np.zeros((3,2), dtype=int),
                          np.block([[
                              np.full((2, 3), 2)],
                                    [np.linspace(-90,-70,3,dtype=int)
                                     ]])
                          ]])
               ]])
    ,
    np.full((5, 1), 10)
    ]])
print(A)
print("\n")

#Zadanie 4
B=A[1,:]+A[3,:]
print(B)
print("\n")

#Zadanie 5
C=np.max(A,1)
print(C)
print("\n")

#Zadanie 6
D=np.delete(np.delete(B, 0), (np.size(np.delete(B, 0))-1))
print(D)
print("\n")

