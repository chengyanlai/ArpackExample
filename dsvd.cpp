#include <iostream>
#include <cassert>
#include <vector>
#include <Eigen/Dense>
#include "arpack.hpp"
#include "lapack/lapack.h"

/*
 *    This code shows how to use ARPACK to find a few of the
 *    largest singular values(sigma) and corresponding right singular
 *    vectors (v) for the the matrix A by solving the symmetri *problem:
 *
 *                       (A'*A)*v = sigma*v
 *
 *    where A is an m by n real matrix.
 *
 *    This code may be easily modified to estimate the 2-norm
 *    condition number  largest(sigma)/smallest(sigma) by setting
 *    which = 'BE' below.  This will ask for a few of the smallest
 *    and a few of the largest singular values simultaneously.
 *    The condition number could then be estimated by taking
 *    the ratio of the largest and smallest singular values.
 *
 *    This formulation is appropriate when  m  .ge.  n.
 *    Reverse the roles of A and A' in the case that  m .le. n.
*/

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
          Eigen::AutoAlign|Eigen::RowMajor> MatrixType;

void mvprod(const size_t &row, const size_t &col, const Eigen::MatrixXd &M,
  double* x, double* y);

int main(int argc, char const *argv[]) {
  int row = atoi(argv[1]);
  int col = atoi(argv[2]);
  MatrixType H = MatrixType::Random(row,col);
  int nev = 4;
  double tol = 0.0e0;
  /*
  // row, col : dimension of the matrix
  // input_ptr: input trail vector pointer
  // nev      : number of singular values to calculate
  // tol      : tolerance. 0 - calculate until machine precision
  */
  int n = row <= col ? row : col;
  /* NCV sets the length of the Arnoldi factorization */
  std::cout << "SVD on a " << row << " x " << col <<
    " random matrix, so we use n = " << n << std::endl;
  int ncv = 24;
  if ( ncv < nev + 1 ) {
    ncv = nev + 1;
  } else if ( ncv > n ){
    ncv = n;
  }
  char bmat  = 'I';
  /* Ask for the NEV singular values of
   * largest magnitude
   *   (indicated by which = 'LM')
   * See documentation in DSAUPD for the
   * other options SM, BE. */
  char which[] = {'L','M'};
  /* IDO  is the REVERSE COMMUNICATION parameter
   *      used to specify actions to be taken on return
   *      from DSAUPD. (See usage below.)
   *
   *      It MUST initially be set to 0 before the first
   *      call to DSAUPD. */
  int ido = 0;
  int lworkl = ncv*(ncv+8);
  int info = 0;

  int *iparam = new int[11];
  iparam[0] = 1;   // Specifies the shift strategy (1->exact)
  iparam[2] = 3*n; // Maximum number of iterations
  iparam[6] = 1;   /* Sets the mode of dsaupd.
                      1 is exact shifting,
                      2 is user-supplied shifts,
                      3 is shift-invert mode,
                      4 is buckling mode,
                      5 is Cayley mode. */
  int *ipntr = new int[11];
  // how many eigenvectors to calculate: 'A' => nev eigenvectors
  char howmny = 'A';
  int *select;
  if( howmny == 'A' )
    select = new int[ncv];
  double* v = new double[col*ncv];
  double* u = new double[row*nev];
  double* workl = new double[lworkl];
  double* workd = new double[3*n];
  double* resid = new double[n];

  dsaupd_(&ido, &bmat, &n, &which[0], &nev, &tol, resid, &ncv, v, &col,
          iparam, ipntr, workd, workl, &lworkl, &info);
  while( ido != 99 ){
    /* Perform matrix vector multiplications
     *              w <--- A*x       (av())
     *              y <--- A'*w      (atv())
     * The user should supply his/her own
     * matrix vector multiplication routines
     * here that takes workd(ipntr(1)) as
     * the input, and returns the result in
     * workd(ipntr(2)). */
    mvprod(row, col, H, workd+ipntr[0]-1, workd+ipntr[1]-1);
    dsaupd_(&ido, &bmat, &n, &which[0], &nev, &tol, resid, &ncv, v, &col,
            iparam, ipntr, workd, workl, &lworkl, &info);
  }
  if( info < 0 )
    std::cerr << "Error with dsaupd, info = " << info << std::endl;
  else if ( info == 1 )
    std::cerr << "Maximum number of Lanczos iterations reached." << std::endl;
  else if ( info == 3 )
    std::cerr << "No shifts could be applied during implicit Arnoldi update," <<
                 "try increasing NCV." << std::endl;
  int rvec = 1;
  double* s = new double[2*ncv];
  double sigma;
  dseupd_(&rvec, &howmny, select, s, v, &col, &sigma, &bmat, &n, which, &nev,
          &tol, resid, &ncv, v, &col, iparam, ipntr, workd, workl, &lworkl, &info);
  if ( info != 0 )
    std::cerr << "Error with dseupd, info = " << info << std::endl;

  std::cout << "Arpack results" << std::endl;
  std::cout << "Converged #" << iparam[4] << std::endl;
  for (ptrdiff_t cnt = nev - 1; cnt >= 0; cnt--) {
    s[cnt] = std::sqrt(s[cnt]);
    std::cout << "S[" << cnt << "]: " << s[cnt] << std::endl;
  }

  delete [] s;
  delete [] resid;
  delete [] workd;
  delete [] workl;
  delete [] u;
  delete [] v;
  if( howmny == 'A' )
    delete [] select;
  delete [] ipntr;
  delete [] iparam;

  std::cout << "Compare to LAPACK" << std::endl;
  double* U2 = new double[row*n];
  double* S2 = new double[n];
  double* vT2 = new double[n*col];
  matrixSVD(H.data(), row, col, U2, S2, vT2);
  for (size_t cnt = 0; cnt < nev; cnt++) {
    std::cout << "S[" << cnt << "]: " << S2[cnt] << std::endl;
  }
}

void mvprod(const size_t &row, const size_t &col, const Eigen::MatrixXd &M,
  double* x, double* y){
  Eigen::VectorXd work;
  if ( row >= col ) {
    Eigen::Map<Eigen::VectorXd> Vx(x, col);
    Eigen::Map<Eigen::VectorXd> Vy(y, col);
    work = M * Vx;
    Vy = M.transpose() * work;
    memcpy(y, Vy.data(), col * sizeof(double) );
  } else {
    Eigen::Map<Eigen::VectorXd> Vx(x, row);
    Eigen::Map<Eigen::VectorXd> Vy(y, row);
    work = M.transpose() * Vx;
    Vy = M * work;
    memcpy(y, Vy.data(), row * sizeof(double) );
  }
}
