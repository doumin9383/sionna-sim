# Matrix inversion

Many signal processing algorithms are described using inverses and
square roots of matrices. However, their actual computation is very
rarely required and one should resort to alternatives in practical
implementations instead. Below, we will describe two representative
examples.

## Solving linear systems

One frequently needs to compute equations of the form

**x** = **H**<sup>−1</sup>**y**

and would be tempted to implement this equation in the following way:

``` python
# Create random example
x_ = tf.random.normal([10, 1])
h = tf.random.normal([10, 10])
y = tf.linalg.matmul(h, x_)

# Solve via matrix inversion
h_inv = tf.linalg.inv(h)
x = tf.linalg.matmul(h_inv, y)
```

A much more stable and efficient implementation avoids the inverse
computation and solves the following linear system instead

**H****x** = **y**

which looks in code like this:

``` python
# Solve as linar system
x = tf.linalg.solve(h, y)
```

When **H** is a Hermitian positive-definite matrix, we can leverage the
[Cholesky
decomposition](https://en.wikipedia.org/wiki/Cholesky_decomposition)
**H** = **L****L**<sup>H</sup>, where **L** is a lower-triangular
matrix, for an even faster implementation:

``` python
# Solve via Cholesky decomposition
l = tf.linalg.cholesky(h)
x = tf.linalg.cholesky_solve(l, y)
```

This is the recommended approach for solving linear systems that we use
throughout Sionna.

## Correlated random vectors

Assume that we need to generate a correlated random vector with a given
covariance matrix, i.e.,

$$\mathbf{x} = \mathbf{R}^{\frac12} \mathbf{w}$$

where **w** ∼ 𝒞𝒩(**0**, **I**) and **R** is known. One should avoid the
explicit computation of the matrix square root here and rather leverage
the Cholesky decomposition for a numerically stable and efficient
implementation. We can compute **R** = **L****L**<sup>H</sup> and then
generate **x** = **L****w**, which can be implemented as follows:

``` python
# Create covariance matrix
r = tf.constant([[1.0, 0.5, 0.25],
                 [0.5, 1.0, 0.5],
                 [0.25, 0.5, 1.0]])

# Cholesky decomposition
l = tf.linalg.cholesky(r)

# Create batch of correlated random vectors
w = tf.random.normal([100, 3])
x = tf.linalg.matvec(l, w)
```

It also happens, that one needs to whiten a correlated noise vector

$$\mathbf{w} = \mathbf{R}^{-\frac12}\mathbf{x}$$

where **x** is random whith covariance matrix
**R** = 𝔼\[**x****x**<sup>H</sup>\]. Rather than computing
$\mathbf{R}^{-\frac12}$, it is sufficient to compute **L**<sup>−1</sup>,
which can be achieved by solving the linear system **L****X** = **I**,
exploiting the diagonal structure of the Cholesky decomposition **L**:

``` python
# Create covariance matrix
r = tf.constant([[1.0, 0.5, 0.25],
                 [0.5, 1.0, 0.5],
                 [0.25, 0.5, 1.0]])

# Cholesky decomposition
l = tf.linalg.cholesky(r)

# Inverse of L
l_inv = tf.linalg.triangular_solve(l, tf.eye(3), lower=True)
```
