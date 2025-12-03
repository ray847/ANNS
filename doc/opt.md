# Optimization

## NaiveSolution

We just calculate the distance of the query vector to every other vector and
select the closest k base vectors.

### Complexity Analysis

Build Time: O(N * d)
Build Space (Final / During): O(N * d) / O(N * d)
Search Time: O(N * d + qlogN)
Search Space: O(N)

## HNSW Optimizations

1. Distance sorting
2. Dynamic exploration factor

## Complexity Analysis

$$\text{Vector Dimension} = d$$
$$|\{\text{Base Vector}\}| = N$$
$$\text{Query Count} = q$$

|Algorithm|Build Time|Build Space (Final / During) |Search Time|Search Space|
|-|-|-|-|-|
|Naive|O(N * d)|O(N * d) / O(N * d)|O(N * d + qlogN)|O(N)|

