#include <iostream>
#include <cassert>
#include <vector>
#include <complex>
#include <stdexcept>
#include <Eigen/Dense>
#include "arpack.hpp"
#include "lapack/lapack.h"

typedef std::complex<double> ComplexType;

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
          Eigen::AutoAlign|Eigen::RowMajor> dMatrixType;

typedef Eigen::Matrix<ComplexType, Eigen::Dynamic, Eigen::Dynamic,
          Eigen::AutoAlign|Eigen::RowMajor> zMatrixType;

typedef Eigen::Matrix<double, Eigen::Dynamic, 1,
          Eigen::AutoAlign> dVectorType;

typedef Eigen::Matrix<ComplexType, Eigen::Dynamic, 1,
          Eigen::AutoAlign> zVectorType;

void arpackSVD(const size_t &r, const size_t &c, const dMatrixType &H,
  const size_t &k, dMatrixType &U, dMatrixType &vT, dMatrixType &SD,
  const double &tolerance = 0.0e0);
void arpackSVD(const size_t &r, const size_t &c, const zMatrixType &H,
  const size_t &k, zMatrixType &U, zMatrixType &vT, dMatrixType &SD,
  const double &tolerance = 0.0e0);

void mvprod(const size_t &row, const size_t &col, const dMatrixType &M,
  double* x, double* y);
void mvprod(const size_t &row, const size_t &col, const zMatrixType &M,
  ComplexType* x, ComplexType* y);

int main(int argc, char const *argv[]) {
  if ( argc < 2 ) {
    throw std::runtime_error("Please input 'row' col' 'nev'");
  }
  std::cout << "Running arpack.app" << std::endl;
  std::cout << std::endl << "Real Type" << std::endl;
  int row = atoi(argv[1]);
  int col = atoi(argv[2]);
  int k = atoi(argv[3]);
  dMatrixType H = dMatrixType::Random(row,col);
  dMatrixType U, S, vT;
  arpackSVD(row, col, H, k, U, vT, S, 0.0e0);
  std::cout << "Compare to LAPACK(left column):" << std::endl;
  int n = row <= col ? row : col;
  double* U2 = new double[row*n];
  double* S2 = new double[n];
  double* vT2 = new double[n*col];
  matrixSVD(H.data(), row, col, U2, S2, vT2);
  Eigen::Map<dMatrixType> Us2(U2, row, n);
  Eigen::Map<dMatrixType> vTs2(vT2, n, col);
  Eigen::Map<dVectorType> Sv(S2, n);
  for (size_t i = 0; i < k; i++) {
    std::cout << std::scientific << Sv(i) << " " << S(i,i) << std::endl;
  }
  std::cout << "check unitary" << std::endl;
  std::cout << "from arpack - U" << std::endl;
  std::cout << U.adjoint() * U << std::endl;
  std::cout << "from arpack - vT" << std::endl;
  std::cout << vT * vT.adjoint() << std::endl;
  std::cout << "from lapack - vT" << std::endl;
  std::cout << vTs2 * vTs2.adjoint() << std::endl;

  std::cout << std::endl << "Complex Type" << std::endl;

  zMatrixType zH = zMatrixType::Random(row,col);
  zMatrixType zU, zvT;
  dMatrixType zS;
  arpackSVD(row, col, zH, k, zU, zvT, zS, 0.0e0);
  std::cout << "Compare to LAPACK(left column)" << std::endl;
  ComplexType* zU2 = new ComplexType[row*n];
  double* zS2 = new double[n];
  ComplexType* zvT2 = new ComplexType[n*col];
  matrixSVD(zH.data(), row, col, zU2, zS2, zvT2);
  Eigen::Map<zMatrixType> zUs2(zU2, row, n);
  Eigen::Map<zMatrixType> zvTs2(zvT2, n, col);
  Eigen::Map<dVectorType> zSv(zS2, n);
  for (size_t i = 0; i < k; i++) {
    std::cout << std::scientific << zSv(i) << " " << zS(i,i) << std::endl;
  }
  std::cout << "check unitary" << std::endl;
  std::cout << "from arpack - U" << std::endl;
  std::cout << zU.adjoint() * zU << std::endl;
  std::cout << "from arpack - vT" << std::endl;
  std::cout << zvT * zvT.adjoint() << std::endl;
  std::cout << "from lapack - vT" << std::endl;
  std::cout << zvTs2 * zvTs2.adjoint() << std::endl;

  delete [] zvT2;
  delete [] zS2;
  delete [] zU2;
  delete [] vT2;
  delete [] S2;
  delete [] U2;
}

void arpackSVD(const size_t &r, const size_t &c, const dMatrixType &H,
  const size_t &k, dMatrixType &U, dMatrixType &vT, dMatrixType &SD,
  const double &tolerance){
  int row = r;
  int col = c;
  int nev = k;
  double tol = tolerance;
  int n = row <= col ? row : col;
  std::cout << "SVD on a " << row << " x " << col <<
    " random matrix, so we use n = " << n << std::endl;
  if ( k > n ) {
    std::cerr << "Number of singular value asked should be smaller than" <<
      " matrix dimension" << std::endl;
  }
  int ncv = 24;
  if ( ncv < nev + 1 ) {
    ncv = (n + nev) / 2;
  } else if ( ncv > n ){
    ncv = n;
  }
  char bmat  = 'I';
  char which[] = {'L','M'};
  int ido = 0;
  int lworkl = ncv*(ncv+8);
  int info = 0;
  int *iparam = new int[11];
  iparam[0] = 1;
  iparam[2] = 3*n;
  iparam[6] = 1;
  int *ipntr = new int[11];
  double* v = new double[n*ncv];
  double* workl = new double[lworkl];
  double* workd = new double[3*n];
  double* resid = new double[n];

  dsaupd_(&ido, &bmat, &n, &which[0], &nev, &tol, resid, &ncv, v, &n,
          iparam, ipntr, workd, workl, &lworkl, &info);
  while( ido != 99 ){
    mvprod(row, col, H, workd+ipntr[0]-1, workd+ipntr[1]-1);
    dsaupd_(&ido, &bmat, &n, &which[0], &nev, &tol, resid, &ncv, v, &n,
            iparam, ipntr, workd, workl, &lworkl, &info);
  }
  if( info < 0 )
    std::cerr << "Error with dsaupd, info = " << info << std::endl;
  else if ( info == 1 )
    std::cerr << "Maximum number of Lanczos iterations reached, and found " <<
              iparam[4] << " converged." << std::endl;
  else if ( info == 3 )
    std::cerr << "No shifts could be applied during implicit Arnoldi update," <<
                 "try increasing NCV." << std::endl;

  int rvec = 1;
  char howmny = 'A';
  int *select = new int[ncv];
  double* s = new double[2*ncv];
  double sigma;
  dseupd_(&rvec, &howmny, select, s, v, &n, &sigma, &bmat, &n, which, &nev,
          &tol, resid, &ncv, v, &n, iparam, ipntr, workd, workl, &lworkl, &info);
  if ( info != 0 )
    std::cerr << "Error with dseupd, info = " << info << std::endl;

  U = dMatrixType::Zero(row,nev);
  SD = dMatrixType::Zero(nev,nev);
  vT = dMatrixType::Zero(nev,col);
  for (ptrdiff_t cnt = nev - 1; cnt >= 0; cnt--) {
    SD(nev - cnt - 1,nev - cnt - 1) = std::sqrt(s[cnt]);
    if ( row >= col ) {
      double* vTa = new double[col];
      memcpy(vTa, &v[col*cnt], col * sizeof(double));
      Eigen::Map<dVectorType> work1(vTa, col);
      vT.row(nev - cnt - 1) = work1;
      dVectorType work2 = H * work1;
      work2.normalize();
      U.col(nev - cnt - 1) = work2;
      delete [] vTa;
    } else {
      double* ua = new double[row];
      memcpy(ua, &v[row*cnt], row * sizeof(double));
      Eigen::Map<dVectorType> work1(ua, row);
      U.col(nev - cnt - 1) = work1;
      dVectorType work2 = H.transpose() * work1;
      work2.normalize();
      vT.row(nev - cnt - 1) = work2;
      delete [] ua;
    }
  }

  delete [] s;
  delete [] select;
  delete [] resid;
  delete [] workd;
  delete [] workl;
  delete [] v;
  delete [] ipntr;
  delete [] iparam;
}

void mvprod(const size_t &row, const size_t &col, const dMatrixType &M,
  double* x, double* y){
  dVectorType work;
  if ( row >= col ) {
    Eigen::Map<dVectorType> Vx(x, col);
    Eigen::Map<dVectorType> Vy(y, col);
    work = M * Vx;
    Vy = M.transpose() * work;
    memcpy(y, Vy.data(), col * sizeof(double) );
  } else {
    Eigen::Map<dVectorType> Vx(x, row);
    Eigen::Map<dVectorType> Vy(y, row);
    work = M.transpose() * Vx;
    Vy = M * work;
    memcpy(y, Vy.data(), row * sizeof(double) );
  }
}

void arpackSVD(const size_t &r, const size_t &c, const zMatrixType &H,
  const size_t &k, zMatrixType &U, zMatrixType &vT, dMatrixType &SD,
  const double &tolerance){
  int row = r;
  int col = c;
  int nev = k;
  double tol = tolerance;
  int n = row <= col ? row : col;
  std::cout << "SVD on a " << row << " x " << col <<
    " random matrix, so we use n = " << n << std::endl;
  int ncv = 24;
  if ( ncv < nev + 1 ) {
    ncv = (nev + n) / 2;
  } else if ( ncv > n ){
    ncv = n;
  }
  char bmat  = 'I';
  char which[] = {'L','R'};
  int ido = 0;
  int lworkl = 3*ncv*(ncv+2);
  int info = 0;

  int *iparam = new int[11];
  iparam[0] = 1;
  iparam[2] = 3*n;
  iparam[6] = 1;
  int *ipntr = new int[14];
  ComplexType* v = new ComplexType[n*ncv];
  ComplexType* workl = new ComplexType[lworkl];
  ComplexType* workd = new ComplexType[3*n];
  ComplexType* resid = new ComplexType[n];
  double *rwork = new double[ncv];

  znaupd_(&ido, &bmat, &n, &which[0], &nev, &tol, resid, &ncv, v, &n,
          iparam, ipntr, workd, workl, &lworkl, rwork, &info);
  while( ido != 99 ){
    mvprod(row, col, H, workd+ipntr[0]-1, workd+ipntr[1]-1);
    znaupd_(&ido, &bmat, &n, &which[0], &nev, &tol, resid, &ncv, v, &n,
            iparam, ipntr, workd, workl, &lworkl, rwork, &info);
  }

  if( info < 0 )
    std::cerr << "Error with znaupd, info = " << info << std::endl;
  else if ( info == 1 )
    std::cerr << "Maximum number of Lanczos iterations reached." << std::endl;
  else if ( info == 3 )
    std::cerr << "No shifts could be applied during implicit Arnoldi update," <<
                 "try increasing NCV." << std::endl;

  int rvec = 1;
  char howmny = 'A';
  int *select = new int[ncv];
  ComplexType* s = new ComplexType[2*ncv];
  ComplexType *workev = new ComplexType[3*ncv];
  ComplexType sigma;
  zneupd_(&rvec, &howmny, select, s, v, &n, &sigma, workev,
          &bmat, &n, &which[0], &nev, &tol, resid, &ncv, v, &n, iparam, ipntr,
          workd, workl, &lworkl, rwork, &info);
  if ( info != 0 )
    std::cerr << "Error with zneupd, info = " << info << std::endl;
  U = zMatrixType::Zero(row,nev);
  SD = dMatrixType::Zero(nev,nev);
  vT = zMatrixType::Zero(nev,col);
  for (size_t cnt = 0; cnt < nev; cnt++) {
    SD(cnt,cnt) = std::sqrt(s[cnt]).real();
    /* NOTE: Calculate left singular vectors */
    if ( row >= col ) {
      ComplexType* vTa = new ComplexType[col];
      memcpy(vTa, &v[col*cnt], col * sizeof(ComplexType));
      Eigen::Map<zVectorType> work1(vTa, col);
      vT.row(cnt) = work1;
      zVectorType work2 = H * work1;
      work2.normalize();
      U.col(cnt) = work2;
      delete [] vTa;
    } else {
      ComplexType* ua = new ComplexType[row];
      memcpy(ua, &v[row*cnt], row * sizeof(ComplexType));
      Eigen::Map<zVectorType> work1(ua, row);
      U.col(cnt) = work1;
      zVectorType work2 = H.adjoint() * work1;
      work2.normalize();
      vT.row(cnt) = work2;
      delete [] ua;
    }
  }

  delete [] workev;
  delete [] s;
  delete [] select;
  delete [] rwork;
  delete [] resid;
  delete [] workd;
  delete [] workl;
  delete [] v;
  delete [] ipntr;
  delete [] iparam;
}

void mvprod(const size_t &row, const size_t &col, const zMatrixType &M,
  ComplexType* x, ComplexType* y){
  zVectorType work;
  if ( row >= col ) {
    Eigen::Map<zVectorType> Vx(x, col);
    Eigen::Map<zVectorType> Vy(y, col);
    work = M * Vx;
    Vy = M.adjoint() * work;
    memcpy(y, Vy.data(), col * sizeof(ComplexType) );
  } else {
    Eigen::Map<zVectorType> Vx(x, row);
    Eigen::Map<zVectorType> Vy(y, row);
    work = M.adjoint() * Vx;
    Vy = M * work;
    memcpy(y, Vy.data(), row * sizeof(ComplexType) );
  }
}
