#include <iostream>
#include <complex>
#include <vector>
#include <Eigen/Sparse>
#include "arpack.hpp"

typedef std::complex<double> ComplexType;
typedef Eigen::Matrix<ComplexType, Eigen::Dynamic, 1, Eigen::AutoAlign> CVector;

void mvprod(const size_t &dim, const Eigen::SparseMatrix<ComplexType> &M,
  ComplexType* x, ComplexType* y, const double &alpha);

int main(int argc, char const *argv[]) {
  int n = 100;
  Eigen::SparseMatrix<ComplexType> H(n,n);
  std::vector<Eigen::Triplet<ComplexType> > Trip;
  H.reserve(2*n);
  for (size_t cnt = 0; cnt < n-1; cnt++) {
    Trip.push_back(Eigen::Triplet<ComplexType>(cnt,cnt+1,ComplexType(-1.0e0,0.0e0)));
    Trip.push_back(Eigen::Triplet<ComplexType>(cnt+1,cnt,ComplexType(-1.0e0,0.0e0)));
  }
  H.setFromTriplets(Trip.begin(), Trip.end());
  int nev = 2;
  double tol = 0;
  /*
  // n        : dimension of the matrix
  // input_ptr: input trail vector pointer
  // nev      : number of eigenvalues to calculate
  // tol      : tolerance. 0 - calculate until machine precision
  */
  // ido: reverse communication parameter, must be zero on first iteration
  int ido = 0;
  // bmat: standard eigenvalue problem A*x=lambda*x
  char bmat = 'I';
  // which: calculate the smallest real part eigenvalue
  char which[] = {'S','R'};
  // resid: the residual vector
  ComplexType *resid = new ComplexType[n];
  // the number of columns in v: the number of lanczos vector
  // generated at each iteration, ncv <= n
  // We use the answer to life, the universe and everything, if possible
  int ncv = 42;
  if( n < ncv )
    ncv = n;
  // v containts the lanczos basis vectors
  int ldv = n;
  ComplexType *v = new ComplexType[ldv*ncv];

  int *iparam = new int[11];
  iparam[0] = 1;   // Specifies the shift strategy (1->exact)
  iparam[2] = 3*n; // Maximum number of iterations
  iparam[6] = 1;   /* Sets the mode of dsaupd.
                      1 is exact shifting,
                      2 is user-supplied shifts,
                      3 is shift-invert mode,
                      4 is buckling mode,
                      5 is Cayley mode. */

  int *ipntr = new int[14];
  /* IPNTR   Integer array of length 14.  (OUTPUT)
             Pointer to mark the starting locations in the WORKD and WORKL
             arrays for matrices/vectors used by the Arnoldi iteration.
             -------------------------------------------------------------
             IPNTR(1): pointer to the current operand vector X in WORKD.
             IPNTR(2): pointer to the current result vector Y in WORKD.
             IPNTR(3): pointer to the vector B * X in WORKD when used in
                       the shift-and-invert mode.
             IPNTR(4): pointer to the next available location in WORKL
                       that is untouched by the program.
             IPNTR(5): pointer to the NCV by NCV upper Hessenberg
                       matrix H in WORKL.
             IPNTR(6): pointer to the  ritz value array  RITZ
             IPNTR(7): pointer to the (projected) ritz vector array Q
             IPNTR(8): pointer to the error BOUNDS array in WORKL.
             IPNTR(14): pointer to the NP shifts in WORKL. See Remark 5 below.
             Note: IPNTR(9:13) is only referenced by zneupd. See Remark 2 below.
             IPNTR(9): pointer to the NCV RITZ values of the
                       original system.
             IPNTR(10): Not Used
             IPNTR(11): pointer to the NCV corresponding error bounds.
             IPNTR(12): pointer to the NCV by NCV upper triangular
                        Schur matrix for H.
             IPNTR(13): pointer to the NCV by NCV matrix of eigenvectors
                        of the upper Hessenberg matrix H. Only referenced by
                        zneupd if RVEC = .TRUE. See Remark 2 below.
        -------------------------------------------------------------*/
  ComplexType *workd = new ComplexType[3*n];
  /* WORKD   Complex*16 work array of length 3*N.  (REVERSE COMMUNICATION)
             Distributed array to be used in the basic Arnoldi iteration
             for reverse communication.  The user should not use WORKD
             as temporary workspace during the iteration !!!!!!!!!!
             See Data Distribution Note below.  */
  int lworkl = 3*ncv*(ncv+2);
  /* LWORKL  Integer.  (INPUT)
             LWORKL must be at least 3*NCV**2 + 5*NCV.*/
  ComplexType *workl = new ComplexType[lworkl];
  /* WORKL   Complex*16 work array of length LWORKL.  (OUTPUT/WORKSPACE)
             Private (replicated) array on each PE or array allocated on
             the front end.  See Data Distribution Note below.*/
  double *rwork = new double[ncv];
  /* RWORK   Double precision  work array of length NCV (WORKSPACE)
             Private (replicated) array on each PE or array allocated on
             the front end. */
  int info = 0;
  /* INFO    Integer.  (INPUT/OUTPUT)
    If INFO .EQ. 0, a randomly initial residual vector is used.
    If INFO .NE. 0, RESID contains the initial residual vector,
                    possibly from a previous run.
    Error flag on output.
    =  0: Normal exit.
    =  1: Maximum number of iterations taken.
          All possible eigenvalues of OP has been found. IPARAM(5)
          returns the number of wanted converged Ritz values.
    =  2: No longer an informational error. Deprecated starting
          with release 2 of ARPACK.
    =  3: No shifts could be applied during a cycle of the
          Implicitly restarted Arnoldi iteration. One possibility
          is to increase the size of NCV relative to NEV.
          See remark 4 below.
    = -1: N must be positive.
    = -2: NEV must be positive.
    = -3: NCV-NEV >= 2 and less than or equal to N.
    = -4: The maximum number of Arnoldi update iteration
        must be greater than zero.
    = -5: WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'
    = -6: BMAT must be one of 'I' or 'G'.
    = -7: Length of private work array is not sufficient.
    = -8: Error return from LAPACK eigenvalue calculation;
    = -9: Starting vector is zero.
    = -10: IPARAM(7) must be 1,2,3.
    = -11: IPARAM(7) = 1 and BMAT = 'G' are incompatible.
    = -12: IPARAM(1) must be equal to 0 or 1.
    = -9999: Could not build an Arnoldi factorization.
             User input error highly likely.  Please
           check actual array dimensions and layout.
             IPARAM(5) returns the size of the current Arnoldi
             factorization.
  */
  /* dneupd parameters
     rvec == 0 : calculate only eigenvalue
     rvec > 0 : calculate eigenvalue and eigenvector */
  int rvec = 1;

  // how many eigenvectors to calculate: 'A' => nev eigenvectors
  char howmny = 'A';

  int *select;
  // when howmny == 'A', this is used as workspace to reorder the eigenvectors
  if( howmny == 'A' )
    select = new int[ncv];

  // This vector will return the eigenvalues from the second routine, dseupd.
  ComplexType *d = new ComplexType[nev+1];

  ComplexType *z = 0;
  if( rvec )
    z = new ComplexType[n*nev];
  /* On exit, if RVEC = .TRUE. and HOWMNY = 'A', then the columns of
       Z represent approximate eigenvectors (Ritz vectors) corresponding
       to the NCONV=IPARAM(5) Ritz values for eigensystem
       A*z = lambda*B*z. */

  // not used if iparam[6] == 1
  ComplexType sigma;
  ComplexType *workev = new ComplexType[3*ncv];
  /*WORKEV  Double precision work array of dimension 3*NCV.  (WORKSPACE)*/

  // first iteration
  znaupd_(&ido, &bmat, &n, &which[0], &nev, &tol, resid, &ncv, v, &ldv,
          iparam, ipntr, workd, workl, &lworkl, rwork, &info);

  while( ido != 99 ){
    mvprod(n, H, workd+ipntr[0]-1, workd+ipntr[1]-1, 0.0e0);
    znaupd_(&ido, &bmat, &n, &which[0], &nev, &tol, resid, &ncv, v, &ldv,
            iparam, ipntr, workd, workl, &lworkl, rwork, &info);
  }

  if( info < 0 )
    std::cerr << "Error with znaupd, info = " << info << std::endl;
  else if ( info == 1 )
    std::cerr << "Maximum number of Lanczos iterations reached." << std::endl;
  else if ( info == 3 )
    std::cerr << "No shifts could be applied during implicit Arnoldi update," <<
                 " try increasing NCV." << std::endl;
  zneupd_(&rvec, &howmny, select, d, z, &ldv, &sigma, workev,
          &bmat, &n, &which[0], &nev, &tol, resid, &ncv, v, &ldv, iparam, ipntr,
          workd, workl, &lworkl, rwork, &info);
  if ( info != 0 )
    std::cerr << "Error with dneupd, info = " << info << std::endl;
  std::cout << "E[0] = " << d[0] << std::endl;
  std::cout << "E[1] = " << d[1] << std::endl;
  delete [] workev;
  if( rvec )
    delete [] z;
  delete [] d;
  if( howmny == 'A' )
    delete [] select;
  delete [] rwork;
  delete [] workl;
  delete [] workd;
  delete [] ipntr;
  delete [] iparam;
  delete [] v;
  delete [] resid;

  return 0;
}

void mvprod(const size_t &dim, const Eigen::SparseMatrix<ComplexType> &M,
  ComplexType* x, ComplexType* y, const double &alpha){
  Eigen::Map<CVector> Vx(x, dim);
  Eigen::Map<CVector> Vy(y, dim);
  Vy = M * Vx + alpha * Vy;
  memcpy(y, Vy.data(), dim * sizeof(ComplexType) );
}
