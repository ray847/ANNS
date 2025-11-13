# Optimization

## NaiveSolution

We just calculate the distance of the query vector to every other vector and
select the closest k base vectors.

### Complexity Analysis

Build Time: O(N * d)
Build Space (Final / During): O(N * d) / O(N * d)
Search Time: O(N * d + qlogN)
Search Space: O(N)

## QuickSolution

### Algorithm

Hyper Parameters:
* s: maximum number of samples to take into consideration when calculation the pivot.

#### Build Phase

```pseudo_code
build(base):
    quick_partion(0, N)

quick_partion(st, ed):
    pivot = select_pivot(st, ed)
    mid = partion(st, ed, pivot)
    quick_partion(st, mid)
    quick_partion(mid, ed)

select_pivot(st, ed) -> Plane:
    transformation = eigenvector
    intercept = transformation * average

partion(st, ed, pivot) -> size_t:
    while st != ed:
        if plane(base[st]) < 0:
            swap(base[st], base[ed])
            ed--
        st++
    return st
```

#### Search Phase

```pseudo_code
search(query):
    Find the partion query is in.
    Look around the W vectors.
```

## Complexity Analysis

$$\text{Vector Dimension} = d$$
$$|\{\text{Base Vector}\}| = N$$
$$\text{Query Count} = q$$

|Algorithm|Build Time|Build Space (Final / During) |Search Time|Search Space|
|-|-|-|-|-|
|Naive|O(N * d)|O(N * d) / O(N * d)|O(N * d + qlogN)|O(N)|

