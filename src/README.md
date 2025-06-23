# AGNI (Analysis of Global Normal-modes in Ideal MHD)

The code needs to have three different parts:

* Class for input and geometry data reading
    * currently only load data from DESC
* Class to create the differentiation matrix structure and the matrix vector product
    * add the differentiation matrices as a separate functions. Two functions: Fourier and Chebyshev
    * add another function where these matrices are assigned and multiplied into larger matrices
    * add a function to convert all the large matrices to float32.
    * then a final function that calculates the matrix vector product
* Class to call the various customized reduced-space eigendecomposition routines
    * add the randomized SVD and block Lanzos as separate functions.
      The same function should probably do eiegen-decomposition.
