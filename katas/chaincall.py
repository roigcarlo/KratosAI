class add(int):
    def __call__(self, *args):
        return self + add(*args)

a = add(1)
print(a) # 1

b = add(1)(2)
print(b) # 3

print(a + b)