#ifndef __LAPACK_H__
#define __LAPACK_H__
#include <complex>

void matrixEigh(double* Kij, int N, double* Eig, double* EigVec);
void matrixEigh(std::complex<double>* Kij, int N, double* Eig, std::complex<double>* EigVec);

void matrixSVD(double* Mij_ori, int M, int N, double* U, double* S, double* vT);
void matrixSVD(std::complex<double>* Mij_ori, int M, int N, std::complex<double>* U, double* S, std::complex<double>* vT);

#endif /* end of include guard: __LAPACK_H__ */
