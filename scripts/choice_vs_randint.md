`randint` is 5x faster!

# On Linux (GCP)

```
In [4]: %%timeit
   ...: np.random.choice(1000000, size=32)
   ...:
13.9 µs ± 113 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

In [5]: %%timeit
   ...: np.random.randint(1000000, size=32)
   ...:
2.57 µs ± 12.9 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

In [7]: %%timeit
   ...: random.choices(range(1000000), k=32)
   ...:
12.9 µs ± 87.4 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
```

# On MacBook Pro

```
In [117]: %%timeit
     ...: np.random.choice(1000000, size=32)
     ...:
12 µs ± 2.22 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)

In [118]: %%timeit
     ...: np.random.randint(1000000, size=32)
     ...:
2.13 µs ± 13.8 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

In [121]: %%timeit
     ...: random.choices(range(1000000), k=32)
     ...:
     ...:
12.5 µs ± 1.36 µs per loop (mean ± std. dev. of 7 runs, 100000 loops each)
```
