#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdexcept>
#include "lapack.h"
#ifdef MKL
  typedef std::complex<double> DCOMP;
  #define MKL_Complex16 DCOMP
  #include "mkl.h"
#else
  #include "lapack_wrapper.h"
#endif

void matrixEigh(double* Kij, int N, double* Eig, double* EigVec){
    memcpy(EigVec, Kij, N * N * sizeof(double));
    int ldA = N;
    int lwork = -1;
    double worktest;
    int info;
    dsyev((char*)"V", (char*)"U", &N, EigVec, &ldA, Eig, &worktest, &lwork, &info);
    if(info != 0){
        std::ostringstream err;
        err << "\nError in Lapack function 'dsyev': Lapack INFO = " << info << "\n";
        throw std::runtime_error(err.str());
    }
    lwork = (int)worktest;
    double* work= (double*)malloc(sizeof(double)*lwork);
    dsyev((char*)"V", (char*)"U", &N, EigVec, &ldA, Eig, work, &lwork, &info);
    if(info != 0){
        std::ostringstream err;
        err << "\n Error in Lapack function 'dsyev': Lapack INFO = " << info << "\n";
        throw std::runtime_error(err.str());
    }
    free(work);
}

void matrixEigh(std::complex<double>* Kij, int N, double* Eig, std::complex<double>* EigVec){
    memcpy(EigVec, Kij, N * N * sizeof(std::complex<double>));
    int ldA = N;
    int lwork = -1;
    std::complex<double> worktest;
    double* rwork = (double*)malloc( (3*N-2) * sizeof(double) );
    int info;
    zheev((char*)"V", (char*)"U", &N, EigVec, &ldA, Eig, &worktest, &lwork, rwork, &info);
    if(info != 0){
        std::ostringstream err;
        err << "\nError in Lapack function 'dsyev': Lapack INFO = " << info << "\n";
        throw std::runtime_error(err.str());
    }
    lwork = (int)worktest.real();
    std::complex<double>* work = (std::complex<double>*)malloc(sizeof(std::complex<double>)*lwork);
    zheev((char*)"V", (char*)"U", &N, EigVec, &ldA, Eig, work, &lwork, rwork, &info);
    if(info != 0){
        std::ostringstream err;
        err << "\n Error in Lapack function 'dsyev': Lapack INFO = " << info << "\n";
        throw std::runtime_error(err.str());
    }
    free(work);
    free(rwork);
}

void matrixSVD(double* Mij_ori, int M, int N, double* U, double* S, double* vT){
    double* Mij = (double*)malloc(M * N * sizeof(double));
    memcpy(Mij, Mij_ori, M * N * sizeof(double));
    int min = M < N ? M : N;    //min = min(M,N)
    int ldA = N, ldu = N, ldvT = min;
    int lwork = -1;
    double worktest;
    int info;
    dgesvd((char*)"S", (char*)"S", &N, &M, Mij, &ldA, S, vT, &ldu, U, &ldvT, &worktest, &lwork, &info);
    if(info != 0){
        std::ostringstream err;
        err<<"\nError in Lapack function 'dgesvd': Lapack INFO = "<<info;
        throw std::runtime_error(err.str());
    }
    lwork = (int)worktest;
    double *work = (double*)malloc(lwork*sizeof(double));
    dgesvd((char*)"S", (char*)"S", &N, &M, Mij, &ldA, S, vT, &ldu, U, &ldvT, work, &lwork, &info);
    if(info != 0){
        std::ostringstream err;
        err<<"\nError in Lapack function 'dgesvd': Lapack INFO = "<<info;
        throw std::runtime_error(err.str());
    }
    free(work);
    free(Mij);
}

void matrixSVD(std::complex<double>* Mij_ori, int M, int N, std::complex<double>* U, double* S, std::complex<double>* vT){
  std::complex<double>* Mij = (std::complex<double>*)malloc(M * N * sizeof(std::complex<double>));
  memcpy(Mij, Mij_ori, M * N * sizeof(std::complex<double>));
  int min = M < N ? M : N;    //min = min(M,N)
  int ldA = N, ldu = N, ldvT = min;
  int lwork = -1;
  double* rwork = (double*)malloc( 5 * min * sizeof(double) );
  int info;
  std::complex<double> worktest;
  zgesvd((char*)"S", (char*)"S", &N, &M, Mij, &ldA, S, vT, &ldu, U, &ldvT, &worktest, &lwork, rwork, &info);
  if(info != 0){
    std::ostringstream err;
    err << "\n1 Error in Lapack function 'zgesvd': Lapack INFO = " << info << " " << ldvT;
    throw std::runtime_error(err.str());
  }
  lwork = (int)(worktest.real());
  std::complex<double> *work = (std::complex<double>*)malloc(lwork*sizeof(std::complex<double>));
  zgesvd((char*)"S", (char*)"S", &N, &M, Mij, &ldA, S, vT, &ldu, U, &ldvT, work, &lwork, rwork, &info);
  if(info != 0){
    std::ostringstream err;
    err << "\n2 Error in Lapack function 'zgesvd': Lapack INFO = " << info << ", lwork = " << lwork;
    throw std::runtime_error(err.str());
  }
  free(work);
  free(rwork);
  free(Mij);
}
