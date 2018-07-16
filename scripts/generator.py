class C:
    def gen(self):
        for i in range(30):
            if i % 2 == 0:
                yield i


if __name__ == '__main__':
    c = C()
    for i in c.gen():
        print(i)

    for i in c.gen():
        print(i)
