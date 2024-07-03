# Time domain methods for the computation of resonances 

by *M. Wess*, *L. Nannen*

TU Wien, Institute of Analysis and Scientific Computing, 


---


This book is designed to provide an introduction and examples to the implementation of time domain methods for the computation of resonances in the high-order finite element library [NGSolve](https://ngsolve.org).


```{card}
**Abstract**
^^^^
To solve resonance problems for time-harmonic wave-type equations without having to invert large system matrices our idea is to utilize fast, explicit time-domain solvers. We compute the eigenvalues of the operator mapping initial values to a filtered time domain solution which can be computed efficiently. This auxiliary eigenvalue problem is solved Krylov space methods. The filter can be designed to map the sought after resonances of the original problem to the largest magnitude resonances of the auxiliary problem while the eigenvectors of the two problems correspond. Stability of the method follows from the stability of the underlying time-domain algorithm.
```

```{note}
For a full mathematical exposition to the method we refer to the extended abstract {cite}`NW24_waves` for a short read and to the preprint {cite}`NW24` for a more comprehensive introduction.
```

---

## Table of Contents
```{tableofcontents}
```


---

## References
```{bibliography}
:filter: docname in docnames
```
