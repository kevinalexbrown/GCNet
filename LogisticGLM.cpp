#include <iostream>
#define ARMA_NO_DEBUG
#include <armadillo>
#include <math.h>


using namespace arma;
using namespace std;

vec cgs( mat A, vec b, vec x0 );

struct Fit 
{
  int iter;
  int iterLim;
  int rangeWarned;
  int weightWarned;
  double dev;
  vec B;
};


// link functions. These follow Shalizi Ch 12 and 13
vec g( vec r )
{
  return log( ( r / (1 - r) ) );
}

vec ig( vec x )
{
  return ( 1 / ( 1 + exp(-x)) );
}

vec dg( vec r )
{
  return ( 1 / (r % (1 - r)) );
}

vec V( vec r )
{
  return ( 1 / (r % (1 - r)) );
}

void printDims( mat X ) 
{
  cout << "nrows: " << X.n_rows << endl;
  cout << "ncols: " << X.n_cols << endl;
}

vec wfit( vec y, mat x, vec w, vec b0 )
{
  //return ( (x.t() * diagmat( w ) * x).i() * x.t() * diagmat( w ) * y );
  //return ( solve( symmatu( x.t() * diagmat( w ) * x ), x.t() * diagmat( w ) * y ) );
  //return ( solve( ( x.t() * diagmat( w ) * x ), x.t() * diagmat( w ) * y ) );

  return( cgs( x.t() * diagmat( w ) * x , x.t() * diagmat( w ) * y , b0 ) ); 
}

vec weights( vec r ) 
{
    //w = 1 / ( abs(1/((r) % (1-r))) % sqrt(r) % sqrt( 1-r) );
    //w = 1 / ( pow( dg( r ), 2 ) % V( r ) );
    return ( r % (1 - r) );
}

double dev( vec r, vec y )
{
//    d = 2.0 * sum( y % log( (y + (1 - y)) / r ) + 
//                     (1 - y) % log( ((1 - y) + y ) / (1 - r) ) );
  return ( 2.0 * sum( y % log( 1 / r ) + (1-y) % log( 1 / (1 - r) ) ) );
}

int checkRRange( vec &r )
{
  bool warned = false;
  double ma = r.max();
  double mi = r.min();
  if ((ma > (1.0 - 2e-16)) || (mi < (0.0 + 2e-16)) ) {
    warned = true;
    for ( int i = 0; i < r.n_elem; i++ ) {
      if ( r(i) < 2e-16 ) 
        r(i) = 0.0 + 2e-16;
      if ( r(i) > (1.0 - 2e-16) )
        r(i) = 1.0 - 2e-16; 
      if ( isnan( r(i) ) )
        r(i) = 0.5;
    }
  }
  if (warned) {
    return 1;
  } else {
    return 0;
  }
}

int checkWeights( vec &w ) 
{
  bool warned = false;
  double wtol = (w.max() * pow( 2.2e-16, (2.0 / 3.0) ));
  if ( w.min() <  wtol ) {
    warned = true;
    cout << "Bad scaling! You have been warned!" << endl; 
    cout << "wtol is: " << wtol << endl;
    cout << "wmin is: " << w.min() << endl;
    cout << "wmax is: " << w.max() << endl;
    for ( int i = 0; i < w.n_elem; i++ )
      if ( w(i) < wtol )
        w(i) = wtol;
  }
  if (warned) {
    return 1;
  } else {
    return 0;
  }

}



//int main( int argc, char ** argv ) 
Fit LogisticGLM( mat x, vec y )
{

  int N = x.n_rows, p = x.n_cols;

  // vectors for regression
  vec eta(N), z(N), w(N), r(N), B(p);
  B = B.zeros();
  B(1) = -5;
  vec B_old = B;

  // init
  r = (y + 0.5) / 2;
  //r = y;
  eta = g( r );
  w = weights( r );
  z = eta + (y - r) % dg( r );
  B = wfit( z, x, w, B );

  // deviance variables
  double D = 1000.0, D_old = 0.0, D_diff = 1000.0, wtol;
  double ma, mi;
  //double m = 1000;

  int rangeWarned = 0;
  int weightWarned = 0;

  // while less than our iterlim and while the improvement in
  // deviance is bigger than our threshold, perform IRLS
  int k = 0, iterLim = 100;
  while ( k < iterLim && ( D_diff > .5 ) ) { // || m > 1e-6 ) ) {

    // (a) Calculate linear predictor of linked data and p (response)
    eta = x * B;
    // response is inverse of link
    r = ig( eta );

    // make sure r is within 0 and 1
    rangeWarned += checkRRange( r );

    // make sure weights are well scaled. Issue a warning if they're not
    weightWarned += checkWeights( w );

    // (b) Find the effective transformed responses
    z = eta + (y - r) % dg( r );

    // (c) Calculate the weights
    w = weights( r );

    // (d) Do a weighted linear regression of z on x with weights w
    // and set B_0, B, to the intercept and slopes of this regression
    B_old = B;
    B = wfit( z, x, w, B );

    // calculate deviance between fitted r and actual y
    D_old = D;
    D = dev( r, y );
    D_diff = abs( D_old - D );

    k++;
//    cout << "iter: " << k << " Deviance: " << D << endl;
    //m = (B - B_old).max();
  }

  cout << "Final Deviance: " << D << endl;

  Fit F;
  F.B   = B;
  F.dev = D;
  F.iter = k;
  F.iterLim = iterLim;
  F.weightWarned = weightWarned;
  F.rangeWarned = rangeWarned;

  return F;
}

  //mat L = chol( x.t() * diagmat( w ) * x );
  //vec yw = x.t() * diagmat( w ) * y;
  //vec btmp = solve( L, yw );
  //vec b = solve( L.t(), btmp );
  //return b;
  //


//  mat U, V;
//  vec s;
//
//  vec yw = y % w;
//  mat xw = x % repmat( w, 1, x.n_cols );
//  svd_econ( U, s, V, xw);
//
//  int cutoff = x.n_cols - 10;
//  for ( int k = cutoff; k < s.n_elem; k++ )
//    s(k) = 0;
//  
//  mat xtmp = U * diagmat( s ) * V.t();
//
//  return ( solve( xtmp, yw ) );
//

//  mat Q = Mat<double>(x.n_rows,x.n_cols); 
//  mat R = Mat<double>(x.n_rows,x.n_cols);
//  vec yw = y % w;
//  mat xw = x % repmat( w, 1, x.n_cols );
//  cout << "wfitting" << endl;
//  qr( Q, R, xw );
//  cout << "R cols: " << R.n_cols << " R rows: " << R.n_rows << endl;
//  vec b = solve( R, Q.t() * yw );
//  cout << "b size: " << b.n_elem << endl;
//  return b;
// 
  //y.print("WTF");

