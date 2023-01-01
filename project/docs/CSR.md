=> Use https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html to prototype first !!!

There are seven available sparse matrix types:
- csc_matrix: Compressed Sparse Column format
- csr_matrix: Compressed Sparse Row format
- bsr_matrix: Block Sparse Row format
- lil_matrix: List of Lists format
- dok_matrix: Dictionary of Keys format
- coo_matrix: COOrdinate format (aka IJV, triplet format)
- dia_matrix: DIAgonal format

https://leimao.github.io/blog/CSR-Sparse-Matrix-Multiplication

![](files/CSR.png)
Image from https://dziganto.github.io/Sparse-Matrices-For-Efficient-Machine-Learning

row offsets or row ptr is number of non-zero accumulated until this row.

Advantages
- Efficient arithmetic operations CSR + CSR, CSR × CSR, etc.
- Efficient row slicing.
- Fast matrix vector products.

Disadvantages
- Slow column slicing operations.
- Changes to the sparsity structure are expensive.

## CSR Matrix Multiplication

- - -

## Others

[Representation and Efficient Computation of Sparse Matrix for Neural Networks](https://www.diva-portal.org/smash/get/diva2:1634980/FULLTEXT01.pdf)

### 2.5.3 Block Compressed Sparse Row Format

