#ifndef __ARPACK_HPP__
#define __ARPACK_HPP__
#include <complex>

extern "C" {
  void znaupd_(int *ido, char *bmat, int *n, char *which,
               int *nev, double *tol, std::complex<double> *resid, int *ncv,
               std::complex<double> *v, int *ldv, int *iparam, int *ipntr,
               std::complex<double> *workd, std::complex<double> *workl,
               int *lworkl, double *rwork, int *info);

  void zneupd_(int *rvec, char *All, int *select, std::complex<double> *d,
               std::complex<double> *z, int *ldz, std::complex<double> *sigma,
               std::complex<double> *workev, char *bmat, int *n, char *which, int *nev,
               double *tol, std::complex<double> *resid, int *ncv,
               std::complex<double> *v,
               int *ldv, int *iparam, int *ipntr, std::complex<double> *workd,
               std::complex<double> *workl, int *lworkl, double *rwork, int *info);

  void dsaupd_(int *ido, char *bmat, int *n, char *which,
            int *nev, double *tol, double *resid, int *ncv,
            double *v, int *ldv, int *iparam, int *ipntr,
            double *workd, double *workl, int *lworkl, int *info);

  void dseupd_(int *rvec, char *All, int *select, double *d,
            double *z, int *ldz, double *sigma,
            char *bmat, int *n, char *which, int *nev,
            double *tol, double *resid, int *ncv, double *v,
            int *ldv, int *iparam, int *ipntr, double *workd,
            double *workl, int *lworkl, int *info);
}

#endif /* end of include guard: __ARPACK_HPP__ */
