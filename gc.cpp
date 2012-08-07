#include <iostream>
#define ARMA_NO_DEBUG
#include <armadillo>
#include <math.h>

using namespace arma;
using namespace std;

struct Fit 
{
  int iter;
  int iterLim;
  int rangeWarned;
  int weightWarned;
  double dev;
  vec B;
};

struct Results
{
  field<vec> Bhat;
  mat weightWarned;
  mat rangeWarned;
  mat iters;
  mat dev;
};

void updateData( Fit f, Results &r, unsigned int k, unsigned int kk)
{
    r.Bhat(k,kk)         = f.B;
    r.weightWarned(k,kk) = f.weightWarned;
    r.rangeWarned(k,kk)  = f.rangeWarned;
    r.iters(k,kk)        = f.iter;
    r.dev(k,kk)          = f.dev;
};

void printDims( mat x );

Fit LogisticGLM( mat x, vec y );

Results getGC( const mat &data, const uvec &neuronLabels )
{

  wall_clock t;
  t.tic();

  int numNeurons = neuronLabels.max();

  Results results;
  results.weightWarned = zeros( numNeurons + 1, numNeurons + 1 );
  results.rangeWarned  = zeros( numNeurons + 1, numNeurons + 1 );
  results.iters        = zeros( numNeurons + 1, numNeurons + 1 );
  results.dev          = zeros( numNeurons + 1, numNeurons + 1 );
  results.Bhat         = field<vec>(numNeurons + 1,numNeurons + 1);

  uvec columnLabels( neuronLabels.n_rows ); 
  for (unsigned int k = 0; k < columnLabels.n_elem; k++ )
    columnLabels(k) = k;


  // for each neuron, compute GC from all other neurons
  for ( unsigned int k = 1; k <= numNeurons; k++ ) {

    cout << "Neuron: " << k << " Checking Full Model"
         << "\tin " << t.toc() << " s \t";

    // init some variables
    uvec fullModelInds = find( columnLabels != k );
    uvec neuronLabelsTmp = neuronLabels.elem( fullModelInds );
    mat x = data.cols( fullModelInds );
    vec y = data.col( k );

    // compute full model
    Fit f = LogisticGLM( x, y );
    updateData( f, results, k, 0 );

    // compute reduced model by removing neurons (and their hist terms) 
    // one at a time
    for ( unsigned int kk = 1; kk <= numNeurons; kk++ ) {
      cout << "Neuron: " << k << " Checking Granger Caused By Neuron " 
           << kk << "\tin " << t.toc() << " s \t";

      uvec inds = find( neuronLabelsTmp != kk );
      Fit f = LogisticGLM( x.cols( inds ), y );
      updateData( f, results, k, kk );
    }
  }

  cout << "This took " << t.toc() << " s" << endl;

  return results;
}

void saveResults( Results results, string saveBase )
{
  // save Bhat
  stringstream ss;
  for ( unsigned int k = 0; k < results.Bhat.n_rows; k++ ) {
    for ( unsigned int kk = 0; kk < results.Bhat.n_cols; kk++ ) {
      ss.str( "" );
      ss << saveBase << "Bhat_";

      if ( kk == 0 ) {
        ss << k << "_Full.dat";
      } else {
        ss << k << "_" << kk << ".dat";
      }

      (results.Bhat( k, kk )).save( ss.str(), raw_ascii );
    }
  }

  results.weightWarned.save( "/home/kevbrown/TmpBeagleCode/weightWarned.dat", raw_ascii );
  results.rangeWarned.save( "/home/kevbrown/TmpBeagleCode/rangeWarned.dat", raw_ascii );
  results.iters.save( "/home/kevbrown/TmpBeagleCode/iters.dat", raw_ascii );
  results.dev.save( "/home/kevbrown/TmpBeagleCode/dev.dat", raw_ascii );
}
   

int main( int argc, char ** argv )
{
  string saveBase = "/home/kevbrown/TmpBeagleCode/Bhat/";

  mat data; 
  data.load( "data.dat", raw_ascii );
  cout << "data Loaded" << endl;
  printDims( data );
  data = join_rows( ones<mat>( data.n_rows, 1 ), data );

  uvec neuronLabels;
  neuronLabels.load( "neurons.dat", raw_ascii );

  Results results = getGC( data, neuronLabels ); 
  
  saveResults( results, saveBase );

  return 0;
}


