import array as arr
a=arr.array('i',[1,2,3])
print("Array before insertion:",end="  ")
for i in range(0,3):
    print(a[i],end=" ")
print()
a.insert(1,4)
print("Array after insertion:",end="  ")
for i in a:
    print(i,end="  ")
print()

b=arr.array('d',[1.5,2.7,3.3])
print("Array before insertion:",end="  ")
for i in range(0,3):
    print(b[i],end="  ")
print()
b.append(4.5)
print("Array after insertion:",end="  ")
for i in b:
    print(i,end="  ")
print()
