#include <iostream>
#define ARMA_NO_DEBUG
#include <armadillo>
#include <complex>
#include <math.h>

using namespace arma;
using namespace std;

vec cgs( mat A, vec b, vec x0 )
{
  
  // input: A * x = b
  //mat A;
  //A.load( "A.dat", raw_ascii );
  //A.print("A=");
  vec x;
  //vec b;
  //b.load( "b.dat", raw_ascii );
  //b.print("b=");
//  vec x0 = 0.5 * ones(b.n_elem); // initial guess

  double tol = 1e-6;
  int m = A.n_rows, n = A.n_cols;
  double relres;
  int maxit = 30000;

  if (m != n ) {
    cout << "A must be a square matrix" << endl;
    return x0;
  }

  if (m != b.n_elem) {
    cout << "b must have the same number of rows as A" << endl;
    return x0;
  }

  if ( tol < 2e-16 ) {
    cout << "tolerance is a bit low, fixing" << endl;
    tol = 2e-16;
  } else if ( tol >= 1 ) {
    cout << " tol is a bit big, fixing" << endl;
    tol = 1 - 2e-16;
  }



  // check for all zeros rhs vector
  double n2b = norm( b, 2 );
  if ( n2b < 2e-16 ) {
    x.zeros();
    int flag = 0;
    relres = 0;
    int iter = 0;
    double resvec = 0;
    return x;
  }

  x = x0;
  int flag = 1; // has method succeeded? 1 is no
  vec xmin = x; // iterate which has min resid so far
  int imin = 0; // iteration at which xmin was computed
  double tolb = tol * n2b; // relative tol
  vec r = b - A * x;
  double normr = norm( r, 2 ); // norm of residual
  double normr_act = normr; // active norm?

  // if initial guess is goot
  if ( normr <= tolb ) {
    int flag = 0;
    relres = normr / n2b;
    int iter = 0;
    double resvec = normr;
    return x;
  }

  vec rt = r; // shadow residual? 
  vec resvec = zeros(maxit+1); // this will store our residuals for each iter
  resvec(1) = normr; // from our first go
  double normrmin = normr; // norm of residual from xmin
  double rho = 1; 
  int stag = 0; // stagnation of method flag
  int moresteps = 0;
  //int maxmsteps = min( floor( n / 50.0 ), 
  int maxmsteps = 5;
  int maxstagsteps = 3;

  // now perform the iterations, finally
  vec q, qh, u, uh, uh1, p, ph, ph1, vh;
  double alpha, rho1, beta, rtvh;
  int iter;
  int warned = 0;

  int k;
  for ( k = 0 ; k < maxit; k++ ) {

    //cout << " iter: " << k + 1 << " of max " << maxit << endl;

    rho1 = rho;
    rho = dot( rt, r );
    if ( rho ==0 || isinf( rho ) ) {
      cout << "rho is 0: " << beta << endl;
      flag = 4;
      break;
    }

    if ( k == 0 ) {
      u = r;
      p = u;
    } else {
      beta = rho / rho1;
      if ( beta ==0 || isinf( beta ) ) {
        cout << "beta is 0: " << beta << endl;
        flag = 4;
        break;
      }
      u = r + beta * q;
      p = u + beta * (q + beta * p);
    }

    ph1 = p;
    ph = p;


    vh = A * ph;
    rtvh = dot( rt, vh );
    if ( rtvh ==0 ) {
      cout << "rtvh is 0: " << alpha << endl;
      flag = 4;
      break;
    } else {
      alpha = rho / rtvh;
    }

    if ( isinf( alpha ) ) {
      cout << "alpha is inf: " << alpha << endl;
      flag = 4;
      break;
    }

    q = u - alpha * vh;

    uh1 = u + q;

    uh = uh1;

    // check for method stagnation
    if ( abs( alpha ) * norm( uh, 2 ) < 2e-16*norm( x, 2 ) ) {
      stag = stag + 1;
    } else {
      stag = 0;
    }

    // form the new iterate
    x = x + alpha * uh;
    qh = A * uh;
    r = r - alpha * qh;
    normr = norm( r, 2 );
    normr_act = normr;
    resvec( k + 1 ) = normr;

    // check for convergence
    //cout << "normr: " << normr << endl;
    if ( normr <= tolb || stag >= maxstagsteps || moresteps ) {
      r = b - A * x;
      normr_act = norm( r, 2 );
      resvec( k + 1 ) = normr_act;
      if ( normr_act <= tolb ) {
        flag = 0;
        iter = k;
        break;
      } else {
        if( stag >= maxstagsteps && moresteps == 0 ) {
          stag = 0;
        }
        moresteps = moresteps + 1;
        if ( moresteps >= maxmsteps ) {
          if ( warned == 0 ) {
            cout << "erm, warming? " << endl;
          }
          flag = 3; 
          iter = k;
          break;
        }
      }
    }


    // update minimum norm quantities
    if ( normr_act < normrmin ) {
      normrmin = normr_act;
      xmin = x;
      imin = k;
    }

    if ( stag >= maxstagsteps ) {
      flag = 3;
      break;
    }

  }

  // return solution is first with minimal residual ? 
  if ( flag == 0 ) {
    relres = normr_act / n2b;
  } else {
    r = b - A * xmin;
    if ( norm( r, 2 ) <= normr_act ) {
      x = xmin;
      iter = imin;
      relres = norm( r, 2 ) / n2b;
    } else {
      iter = k;
      relres = normr_act / n2b;
    }
  }

  if ( flag > 0 ) {
    cout << "with flag " << flag << endl;
  }

  //x.print("x = ");
  //xmin.print("xmin = ");
  return xmin;
}


//   // truncate trailing zeros from resvec
//   if ( flag <= 1 || flag == 3 ) {
//     resvec = resvec.cols( 0, k );
//   } else {
//     resvec = resvec.cols( 0, k-1 );
//   }


