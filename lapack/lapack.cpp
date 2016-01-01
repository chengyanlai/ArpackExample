#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "lapack.h"
#ifdef MKL
  typedef std::complex<double> DCOMP;
  #define MKL_Complex16 DCOMP
  #include "mkl.h"
#else
  #include "lapack_wrapper.h"
#endif

void matrixMul(double* A, double* B, int M, int N, int K, double* C)
{
    double alpha = 1, beta = 0;
    dgemm((char*)"N", (char*)"N", &N, &M, &K, &alpha, B, &N, A, &K, &beta, C, &N);
}

void diagMM(double* diag, double* mat, size_t M, size_t N)
{
    for(size_t i = 0; i < M; i++)
        vectorScal(diag[i], &(mat[i * N]), N);
}

// void vectorAdd(const double a, double* Y, double* X, size_t N){    // Y = Y + a * X
void vectorAdd(double* Y, double* X, size_t N)
{    // Y = Y + X
    double a = 1.0;
    int inc = 1;
    int64_t left = N;
    size_t offset = 0;
    int chunk;
    while(left > 0){
        if(left > INT_MAX)
            chunk = INT_MAX;
        else
            chunk = left;
        daxpy(&chunk, &a, X + offset, &inc, Y + offset, &inc);
        offset += chunk;
        left -= INT_MAX;
    }
}

void vectorScal(double a, double* X, size_t N)
{
    int inc = 1;
    int64_t left = N;
    size_t offset = 0;
    int chunk;
    while(left > 0){
        if(left > INT_MAX)
            chunk = INT_MAX;
        else
            chunk = left;
        dscal(&chunk, &a, X + offset, &inc);
        offset += chunk;
        left -= INT_MAX;
    }
}

double vectorSum(double* X, size_t N, int inc)
{
    double sum = 0;
    size_t idx = 0;
    for(size_t i = 0; i < N; i++){
        sum += X[idx];
        idx += inc;
    }
    return sum;
}

double vectorNorm(double* X, size_t N, int inc)
{
    double norm2 = 0;
    double tmp = 0;
    int64_t left = N;
    size_t offset = 0;
    int chunk;
    while(left > 0)
    {
        if(left > INT_MAX)
            chunk = INT_MAX;
        else
            chunk = left;
        tmp = dnrm2(&chunk, X + offset, &inc);
        norm2 += tmp * tmp;
        offset += chunk;
        left -= INT_MAX;
    }
    return sqrt(norm2);
}

void vectorExp(double a, double* X, size_t N)
{
    for(size_t i = 0; i < N; i++)
        X[i] = exp(a * X[i]);
}

void syDiag(double* Kij, int N, double* Eig, double* EigVec)
{
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

void orthoRandomize(double* elem, int M, int N)
{
    int eleNum = M*N;
    double *random = (double*)malloc(eleNum * sizeof(double));
    for(size_t cnt = 0; cnt < eleNum; cnt++){
        random[cnt] = ((double)rand()) / (double)RAND_MAX;
    }
    // elemRand(random, M * N, false);
    int min = M < N ? M : N;
    double *S = (double*)malloc(min*sizeof(double));
    if(M <= N){
        double *U = (double*)malloc(M * min * sizeof(double));
        matrixSVD(random, M, N, U, S, elem);
        free(U);
    }
    else{
        double *VT = (double*)malloc(min * N * sizeof(double));
        matrixSVD(random, M, N, elem, S, VT);
        free(VT);
    }
    free(random);
    free(S);
}

void matrixSVD(double* Mij_ori, int M, int N, double* U, double* S, double* vT)
{
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

void matrixSVD(std::complex<double>* Mij_ori, int M, int N, std::complex<double>* U, double* S, std::complex<double>* vT)
{
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
