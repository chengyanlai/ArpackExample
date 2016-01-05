#include <iostream>
#include <vector>
#include <Eigen/Sparse>
#include "arpack.hpp"
#include "lapack/lapack.h"

void mvprod(const size_t &dim, const Eigen::SparseMatrix<double> &M,
  double* x, double* y, const double &alpha);

int main(int argc, char const *argv[]) {
  int n = 100;
  Eigen::SparseMatrix<double> H(n,n);
  std::vector<Eigen::Triplet<double> > Trip;
  H.reserve(2*n);
  for (size_t cnt = 0; cnt < n-1; cnt++) {
    Trip.push_back(Eigen::Triplet<double>(cnt,cnt+1,-1.0e0));
    Trip.push_back(Eigen::Triplet<double>(cnt+1,cnt,-1.0e0));
  }
  H.setFromTriplets(Trip.begin(), Trip.end());
  int nev = 2;
  double tol = 0;
  int ido = 0;
  char bmat = 'I';
  char which[] = {'S','A'};
  double *resid = new double[n];
  int ncv = 42;
  if( n < ncv )
    ncv = n;
  int ldv = n;
  double *v = new double[ldv*ncv];
  int *iparam = new int[11];
  iparam[0] = 1;
  iparam[2] = 3*n;
  iparam[6] = 1;
  int *ipntr = new int[11];
  double *workd = new double[3*n];
  int lworkl = ncv*(ncv+8);
  double *workl = new double[lworkl];
  int info = 0;// random initial
  int rvec = 1;
  char howmny = 'A';
  int *select;
  if( howmny == 'A' )
    select = new int[ncv];
  double *d = new double[nev];
  double *z = 0;
  if( rvec )
    z = new double[n*nev];
  double sigma;
  dsaupd_(&ido, &bmat, &n, &which[0], &nev, &tol, resid, &ncv, v, &ldv,
          iparam, ipntr, workd, workl, &lworkl, &info);
  while( ido != 99 ){
    mvprod(n, H, workd+ipntr[0]-1, workd+ipntr[1]-1, 0.0e0);
    dsaupd_(&ido, &bmat, &n, &which[0], &nev, &tol, resid, &ncv, v, &ldv,
            iparam, ipntr, workd, workl, &lworkl, &info);
  }
  if( info < 0 )
    std::cerr << "Error with dsaupd, info = " << info << std::endl;
  else if ( info == 1 )
    std::cerr << "Maximum number of Lanczos iterations reached." << std::endl;
  else if ( info == 3 )
    std::cerr << "No shifts could be applied during implicit Arnoldi update," <<
                 "try increasing NCV." << std::endl;
  dseupd_(&rvec, &howmny, select, d, z, &ldv, &sigma, &bmat, &n, which, &nev,
          &tol, resid, &ncv, v, &ldv, iparam, ipntr, workd, workl, &lworkl, &info);
  if ( info != 0 )
    std::cerr << "Error with dseupd, info = " << info << std::endl;
  std::cout << "Arpack result" << std::endl;
  std::cout << "E[0]: " << d[0] << std::endl;
  std::cout << "E[1]: " << d[1] << std::endl;

  delete [] resid;
  delete [] v;
  delete [] iparam;
  delete [] ipntr;
  delete [] workd;
  delete [] workl;
  delete [] d;
  if( rvec )
    delete [] z;
  if( howmny == 'A' )
    delete [] select;

  std::cout << "Compare to LAPCK" << std::endl;
  Eigen::MatrixXd Hd(H);
  double *Eig = new double[n];
  double *EigVec = new double[n*n];
  matrixEigh(Hd.data(), n, Eig, EigVec);
  std::cout << "E[0]: " << Eig[0] << std::endl;
  std::cout << "E[1]: " << Eig[1] << std::endl;
  delete [] EigVec;
  delete [] Eig;
  
  return 0;
}

void mvprod(const size_t &dim, const Eigen::SparseMatrix<double> &M,
  double* x, double* y, const double &alpha){
  Eigen::Map<Eigen::VectorXd> Vx(x, dim);
  Eigen::Map<Eigen::VectorXd> Vy(y, dim);
  Vy = M * Vx + alpha * Vy;
  memcpy(y, Vy.data(), dim * sizeof(double) );
}
