//=================================================================================================
/*!
//  \file Sandbox.cpp
//  \brief Sandbox for various Blaze test scenarios
//
//  Copyright (C) 2011 Klaus Iglberger - All Rights Reserved
//
//  This file is part of the Blaze library. This library is free software; you can redistribute
//  it and/or modify it under the terms of the GNU General Public License as published by the
//  Free Software Foundation; either version 3, or (at your option) any later version.
//
//  This library is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
//  without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
//  See the GNU General Public License for more details.
//
//  You should have received a copy of the GNU General Public License along with a special
//  exception for linking and compiling against the Blaze library, the so-called "runtime
//  exception"; see the file COPYING. If not, see http://www.gnu.org/licenses/.
*/
//=================================================================================================


//*************************************************************************************************
// Includes
//*************************************************************************************************

#include <algorithm>
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>
#include <typeinfo>
#include <vector>

#include <blaze/Math.h>
#include <blaze/Util.h>

//#include <boost/numeric/ublas/io.hpp>
//#include <boost/numeric/ublas/matrix.hpp>
//#include <boost/numeric/ublas/matrix_sparse.hpp>
//#include <boost/numeric/ublas/vector.hpp>
//#include <boost/numeric/ublas/vector_sparse.hpp>

//#include <blitz/array.h>

//#include <Eigen/Dense>
//#include <Eigen/Sparse>

//#include <boost/numeric/mtl/mtl.hpp>

//#include <armadillo>

//#include <gmm/gmm.h>

using namespace blaze;




//=================================================================================================
//
//  MAIN FUNCTION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The main function for the Blaze test scenarios.
//
// \return 0 in case of a successful execution, 1 in case of errors.
*/
int main()
{
   // OPERATION TEST //////////////////////////////////////////////////////////////////////////////
   /*
   //setSeed( 306732 );

   // Dense vectors
   StaticVector<int,2UL,columnVector> d1;
   d1[0] = 1;
   d1[1] = 2;
//    d1[2] = 3;
//    d1[3] = 4;
//    d1[4] = 5;
//    d1[5] = 6;

   DynamicVector<int,columnVector> d2( 2UL );
   d2[0] = 1;
   d2[1] = 2;
//    d2[2] = 3.0;
//    d2[3] = 4.0;
//    d2[4] = 5.0;
//    d2[5] = 6.0;

   // Sparse vectors
   CompressedVector<int,rowVector> s1( 2 );
   s1[1] = 2;

   CompressedVector<double,columnVector> s2( 6 );
   s2[0] = 1.0;
   s2[3] = 4.0;

   // Dense matrices
   StaticMatrix<int,6UL,6UL,rowMajor> D1;
   D1(0,0) = 1.0;
   D1(0,2) = 2.0;
   D1(0,4) = 3.0;
   D1(1,1) = 4.0;
   D1(1,3) = 5.0;
   D1(1,5) = 6.0;
   D1(2,0) = 7.0;
   D1(2,2) = 8.0;
   D1(2,4) = 9.0;
   D1(3,1) = 1.0;
   D1(3,3) = 2.0;
   D1(3,5) = 3.0;
   D1(4,0) = 4.0;
   D1(4,2) = 5.0;
   D1(4,4) = 6.0;
   D1(5,1) = 7.0;
   D1(5,3) = 8.0;
   D1(5,5) = 9.0;

   DynamicMatrix<int,columnMajor> D2( 6UL, 6UL, 0 );
   D2(0,0) = 1.0;
   D2(0,2) = 2.0;
   D2(0,4) = 3.0;
   D2(1,1) = 4.0;
   D2(1,3) = 5.0;
   D2(1,5) = 6.0;
   D2(2,0) = 7.0;
   D2(2,2) = 8.0;
   D2(2,4) = 9.0;
   D2(3,1) = 1.0;
   D2(3,3) = 2.0;
   D2(3,5) = 3.0;
   D2(4,0) = 4.0;
   D2(4,2) = 5.0;
   D2(4,4) = 6.0;
   D2(5,1) = 7.0;
   D2(5,3) = 8.0;
   D2(5,5) = 9.0;

   // Sparse matrices
   CompressedMatrix<int,columnMajor> S1( 6, 6 );
   S1(0,0) = 1;
   S1(0,2) = 2;
   S1(0,4) = 3;
   S1(1,1) = 4;
   S1(1,3) = 5;
   S1(1,5) = 6;
   S1(2,0) = 7;
   S1(2,2) = 8;
   S1(2,4) = 9;
   S1(3,1) = 1;
   S1(3,3) = 2;
   S1(3,5) = 3;
   S1(4,0) = 4;
   S1(4,2) = 5;
   S1(4,4) = 6;
   S1(5,1) = 7;
   S1(5,3) = 8;
   S1(5,5) = 9;

   CompressedMatrix<double,rowMajor> S2( 6, 6 );
   S2(0,0) = 1.0;
   S2(0,2) = 2.0;
   S2(1,1) = 3.0;
   S2(1,3) = 4.0;
   S2(2,0) = 5.0;
   S2(2,2) = 6.0;
   S2(3,1) = 7.0;
   S2(3,3) = 8.0;

   // Target vectors
   DynamicVector<int,columnVector> x( 2UL );
   x[0] = 1;
   x[1] = 2;
//    x[2] = 3;
//    x[3] = 4;
//    x[4] = 5;
//    x[5] = 6;

   // Target matrices
   CompressedMatrix<int,rowMajor> X( 6, 6 );
//    X(0,0) = 1.0;
//    X(0,2) = 2.0;
//    X(1,1) = 3.0;
//    X(1,3) = 4.0;
//    X(2,0) = 5.0;
//    X(2,2) = 6.0;
//    X(3,1) = 7.0;
//    X(3,3) = 8.0;

   // Operations
   x = d1 + d2;
   std::cerr << "\n result =\n" << x << "\n\n";

   // Single element access
//    std::cout << " x[0] = " << ( S1 * d1 )[0] << "\n";
//    std::cout << " x[1] = " << ( S1 * d1 )[1] << "\n";
//    std::cout << " x[2] = " << ( S1 * d1 )[2] << "\n";
//    std::cout << " x[3] = " << ( S1 * d1 )[3] << "\n";
//    std::cout << " D(0,1) = " << ( ( D1 * D2 ) * 2.0 )(0,1) << "\n";
//    std::cout << " D(0,2) = " << ( ( D1 * D2 ) * 2.0 )(0,2) << "\n";
//    std::cout << " D(0,3) = " << ( ( D1 * D2 ) * 2.0 )(0,3) << "\n";
//    std::cout << " D(1,0) = " << ( ( D1 * D2 ) * 2.0 )(1,0) << "\n";
//    std::cout << " D(1,1) = " << ( ( D1 * D2 ) * 2.0 )(1,1) << "\n";
//    std::cout << " D(1,2) = " << ( ( D1 * D2 ) * 2.0 )(1,2) << "\n";
//    std::cout << " D(1,3) = " << ( ( D1 * D2 ) * 2.0 )(1,3) << "\n";
//    std::cout << " D(2,0) = " << ( ( D1 * D2 ) * 2.0 )(2,0) << "\n";
//    std::cout << " D(2,1) = " << ( ( D1 * D2 ) * 2.0 )(2,1) << "\n";
//    std::cout << " D(2,2) = " << ( ( D1 * D2 ) * 2.0 )(2,2) << "\n";
//    std::cout << " D(2,3) = " << ( ( D1 * D2 ) * 2.0 )(2,3) << "\n";
//    std::cout << " D(3,0) = " << ( ( D1 * D2 ) * 2.0 )(3,0) << "\n";
//    std::cout << " D(3,1) = " << ( ( D1 * D2 ) * 2.0 )(3,1) << "\n";
//    std::cout << " D(3,2) = " << ( ( D1 * D2 ) * 2.0 )(3,2) << "\n";
//    std::cout << " D(3,3) = " << ( ( D1 * D2 ) * 2.0 )(3,3) << "\n";

//    std::cout << " d[0] = " << ( eval(A) * eval(a) )[0] << "\n";
//    std::cout << " d[1] = " << ( eval(A) * eval(a) )[1] << "\n";
//    std::cout << " d[2] = " << ( eval(A) * eval(a) )[2] << "\n";
//    std::cout << " d[3] = " << ( eval(A) * eval(a) )[3] << "\n";
   */

   /*
   complex* A = new complex[4];
   A[0] = complex( 1.0F, 0.0F );
   A[1] = complex( 2.0F, 0.0F );
   A[2] = complex( 3.0F, 0.0F );
   A[3] = complex( 4.0F, 0.0F );

   complex* B = new complex[4];
   B[0] = complex( 1.0F, 0.0F );
   B[1] = complex( 2.0F, 0.0F );
   B[2] = complex( 3.0F, 0.0F );
   B[3] = complex( 4.0F, 0.0F );

   complex* C = new complex[4];

   const complex alpha( 1.0F, 0.0F );
   const complex beta ( 0.0F, 0.0F );

   cblas_cgemm( CblasRowMajor, CblasNoTrans, CblasNoTrans, 2, 2, 2, &alpha,
                A, 2, B, 2, &beta, C, 2 );

   std::cerr << "\n";
   for( size_t i=0UL; i<2UL; ++i ) {
      for( size_t j=0UL; j<2UL; ++j ) {
         std::cerr << " " << C[i*2+j];
      }
      std::cerr << "\n";
   }
   std::cerr << "\n";
   */

   /*
   // CG-algorithm
   const size_t N( 1000UL );  // Number of rows and columns => N^2 is the number of unknowns
   const size_t maxIterations( 10UL );


   const size_t NN( N*N );
   blaze::timing::WcTimer timer;

   timer.start();
   std::vector<size_t> tmp( NN );
   for( size_t i=0UL; i<N; ++i ) {
      for( size_t j=0UL; j<N; ++j ) {
         size_t nnz( 5UL );
         if( i == 0UL || i == N-1UL ) --nnz;
         if( j == 0UL || j == N-1UL ) --nnz;
         tmp[i*N+j] = nnz;
      }
   }

   blaze::CompressedMatrix<double,rowMajor> A( NN, NN, tmp );
   for( size_t i=0UL; i<N; ++i ) {
      for( size_t j=0UL; j<N; ++j ) {
         if( i > 0UL   ) A.append( i*N+j, (i-1UL)*N+j, -1.0 );  // Top neighbor
         if( j > 0UL   ) A.append( i*N+j, i*N+j-1UL  , -1.0 );  // Left neighbor
         A.append( i*N+j, i*N+j, 4.0 );
         if( j < N-1UL ) A.append( i*N+j, i*N+j+1UL  , -1.0 );  // Right neighbor
         if( i < N-1UL ) A.append( i*N+j, (i+1UL)*N+j, -1.0 );  // Bottom neighbor
      }
   }
   timer.end();
   std::cerr << "\n Setup of matrix A: runtime = " << timer.last() << "\n\n";

   //std::cout << "\n A =\n" << A << "\n\n";
   return 0;

   blaze::DynamicVector<double,false> x( NN );
   for( size_t i=0UL; i<NN; ++i ) {
      x[i] = blaze::rand<double>( -2.0, 2.0 );
   }

   blaze::DynamicVector<double,false> b( N*N, 0.0 );


   //=== CG algorithm ===
   blaze::DynamicVector<double,false> r( NN ), d( NN ), h( NN );
   double alpha, beta, delta, lastPrecision;
   bool converged( false );

   // Computing the initial residual
   r = A * x + b;

   // Initial convergence test
   lastPrecision = 0.0;
   for( size_t i=0UL; i<NN; ++i ) {
      lastPrecision = max( lastPrecision, std::fabs( r[i] ) );
   }

   if( lastPrecision < 1E-8 )
      converged = true;

   delta = trans(r) * r;

   d = -r;

   // Performing the CG iterations
   size_t it( 0UL );

   for( ; !converged && it<maxIterations; ++it )
   {
      h = A * d;

      alpha = delta / ( trans(d) * h );

      x += alpha * d;
      r += alpha * h;

      lastPrecision = 0.0;
      for( size_t i=0UL; i<N; ++i ) {
         lastPrecision = max( lastPrecision, std::fabs( r[i] ) );
      }

      if( lastPrecision < 1E-8 ) {
         converged = true;
         break;
      }

      beta = trans(r) * r;

      d = ( beta / delta ) * d - r;

      delta = beta;
   }
   */
   // END OPERATION TEST //////////////////////////////////////////////////////////////////////////



   // BOOST UBLAS TESTS ///////////////////////////////////////////////////////////////////////////
   /*
   using ::boost::numeric::ublas::row_major;

   const size_t N( 1000UL );

   const size_t NN( N*N );

   boost::numeric::ublas::compressed_matrix<real,row_major> A( NN, NN );
   blaze::timing::WcTimer timer;

   timer.start();
   for( size_t i=0UL; i<N; ++i ) {
      for( size_t j=0UL; j<N; ++j ) {
         if( i > 0UL   ) A(i*N+j,(i-1UL)*N+j) = -1.0;  // Top neighbor
         if( j > 0UL   ) A(i*N+j,i*N+j-1UL  ) = -1.0;  // Left neighbor
         A(i*N+j,i*N+j) = 4.0;
         if( j < N-1UL ) A(i*N+j,i*N+j+1UL  ) = -1.0;  // Right neighbor
         if( i < N-1UL ) A(i*N+j,(i+1UL)*N+j) = -1.0;  // Bottom neighbor
      }
   }
   timer.end();
   std::cerr << "\n Setup of matrix A: runtime = " << timer.last() << "\n\n";
   */
   // END BOOST UBLAS TESTS ///////////////////////////////////////////////////////////////////////



   // BLITZ TESTS /////////////////////////////////////////////////////////////////////////////////
   /*
   ::blitz::Array<real,2> A( 3, 3, ::blitz::fortranArray );
   ::blitz::Array<real,2> B( 3, 3, ::blitz::fortranArray );
   ::blitz::Array<real,1> a( 3 ), b( 3 );
   ::blitz::firstIndex i;
   ::blitz::secondIndex j;
   ::blitz::thirdIndex k;
   ::blaze::timing::WcTimer timer;

   A(1,1) = 1.0;
   A(1,2) = 0.0;
   A(1,3) = 2.0;
   A(2,1) = 0.0;
   A(2,2) = 3.0;
   A(2,3) = 0.0;
   A(3,1) = 4.0;
   A(3,2) = 0.0;
   A(3,3) = 5.0;
   std::cout << "\n A =\n" << A << "\n";

   B(1,1) = 1.0;
   B(1,2) = 0.0;
   B(1,3) = 2.0;
   B(2,1) = 0.0;
   B(2,2) = 3.0;
   B(2,3) = 0.0;
   B(3,1) = 4.0;
   B(3,2) = 0.0;
   B(3,3) = 5.0;
   std::cout << "\n B =\n" << B << "\n";

   a(0) = 1.0;
   a(1) = 2.0;
   a(2) = 3.0;
   std::cout << "\n a =\n" << a << "\n";

   //b = sum( A(i,j) * a(j), j );
   //std::cout << "\n b =\n" << b << "\n\n";

   //::blitz::Array<real,2> C( 3, 3, ::blitz::fortranArray );
   //C = sum( A(i,k) * B(k,j), k );
   //std::cout << "\n C =\n" << C << "\n";
   */
   // END BLITZ TESTS /////////////////////////////////////////////////////////////////////////////



   // ARMADILLO TESTS /////////////////////////////////////////////////////////////////////////////
   /*
   arma::Mat<double> A(3,3);
   A(0,0) = 1.0;
   A(0,1) = 0.0;
   A(0,2) = 2.0;
   A(1,0) = 0.0;
   A(1,1) = 3.0;
   A(1,2) = 0.0;
   A(2,0) = 4.0;
   A(2,1) = 0.0;
   A(2,2) = 5.0;

   arma::Row<double> a( 3 );
   a[0] = 1.0;
   a[1] = 2.0;
   a[2] = 3.0;

   arma::Row<double> b( 3 );
   b[0] = 1.0;
   b[1] = 2.0;
   b[2] = 3.0;

   b = a * A;
   std::cout << "\n b =\n" << b << "\n";
   */
   // END ARMADILLO TESTS /////////////////////////////////////////////////////////////////////////



   // GMM TESTS ///////////////////////////////////////////////////////////////////////////////////
   /*
   std::vector<double> d1( 3 );
   d1[0] = 1.0;
   d1[1] = 2.0;
   d1[2] = 3.0;

   std::vector<double> d2( 3 );
   d2 = d1;

   gmm::rsvector<double> s1( 3 );
   s1[0] = 1.0;
   s1[1] = 2.0;
   s1[2] = 3.0;

   gmm::rsvector<double> s2( 3 );
   s2[0] = 1.0;
   s2[1] = 2.0;
   s2[2] = 3.0;

   gmm::dense_matrix<double> D1( 3, 3 );
   D1(0,0) = 1.0;
   D1(0,1) = 0.0;
   D1(0,2) = 2.0;
   D1(1,0) = 0.0;
   D1(1,1) = 3.0;
   D1(1,2) = 0.0;
   D1(2,0) = 4.0;
   D1(2,1) = 0.0;
   D1(2,2) = 5.0;

   //gmm::dense_matrix<double> D2( D1 );

   gmm::row_matrix< gmm::wsvector<double> > Tmp1( 3, 3 );
   Tmp1(0,0) = 1.0;
   Tmp1(0,2) = 2.0;
   Tmp1(1,1) = 3.0;
   Tmp1(2,0) = 4.0;
   Tmp1(2,2) = 5.0;
   gmm::csr_matrix<double> S1( 3, 3 );
   gmm::copy( Tmp1, S1 );

   gmm::col_matrix< gmm::wsvector<double> > Tmp2( 3, 3 );
   Tmp2(0,0) = 1.0;
   Tmp2(0,2) = 2.0;
   Tmp2(1,1) = 3.0;
   Tmp2(2,0) = 4.0;
   Tmp2(2,2) = 5.0;
   gmm::csc_matrix<double> S2( 3, 3 );
   gmm::copy( Tmp2, S2 );

   std::vector<double> x( 3 );

   gmm::dense_matrix<double> X( 3, 3 );

   //::gmm::mult( d1, D1, x );
   //std::cerr << "\n result =\n" << x << "\n";
   double res = ::gmm::vect_sp( s1, s2 );
   std::cerr << "\n result =\n" << res << "\n";
   */

   /*
   // Conjugate gradient algorithm

   const size_t N( 1000UL );

   const size_t NN( N*N );

   ::gmm::row_matrix< ::gmm::wsvector<double> > T( NN, NN );
   ::gmm::csr_matrix<real> A( NN, NN );
   ::std::vector<real> x( NN ), b( NN ), r( NN ), d( NN ), h( NN ), init( NN );
   real alpha, beta, delta;
   blaze::timing::WcTimer timer;

   timer.start();
   for( size_t i=0UL; i<N; ++i ) {
      for( size_t j=0UL; j<N; ++j ) {
         if( i > 0UL   ) T(i*N+j,(i-1UL)*N+j) = -1.0;  // Top neighbor
         if( j > 0UL   ) T(i*N+j,i*N+j-1UL  ) = -1.0;  // Left neighbor
         T(i*N+j,i*N+j) = 4.0;
         if( j < N-1UL ) T(i*N+j,i*N+j+1UL  ) = -1.0;  // Right neighbor
         if( i < N-1UL ) T(i*N+j,(i+1UL)*N+j) = -1.0;  // Bottom neighbor
      }
   }

   copy( T, A );
   timer.end();
   std::cerr << "\n Setup of matrix A: runtime = " << timer.last() << "\n\n";
   return 0;

   for( size_t i=0UL; i<NN; ++i ) {
      b[i]    = real(0);
      init[i] = ::blaze::rand<real>();
   }

   x = init;
   mult( A, x, r );
   ::gmm::add( b, r );
   delta = ::gmm::vect_sp( r, r );
   copy( ::gmm::scaled( r, real(-1) ), d );

   for( size_t iteration=0UL; iteration<NN; ++iteration )
   {
      mult( A, d, h );
      alpha = delta / ::gmm::vect_sp( d, h );
      ::gmm::add( ::gmm::scaled( d, alpha ), x, x );
      ::gmm::add( ::gmm::scaled( h, alpha ), r, r );
      beta = ::gmm::vect_sp( r, r );
      if( std::sqrt( beta ) < 1E-8 ) break;
      ::gmm::add( ::gmm::scaled( r, real(-1) ), ::gmm::scaled( d, beta / delta ), d );
      delta = beta;
   }

   std::cout << "\n Solution vector x =\n" << x << "\n\n";
   */
   // END GMM TESTS ///////////////////////////////////////////////////////////////////////////////



   // MTL TESTS ///////////////////////////////////////////////////////////////////////////////////
   /*
//    typedef mtl::tag::row_major  row_major;
//    typedef mtl::tag::col_major  col_major;
//    typedef mtl::matrix::parameters<row_major>  row_parameters;
//    typedef mtl::matrix::parameters<col_major>  col_parameters;
//    typedef mtl::compressed2D<real,row_parameters>  row_compressed2D;
//    typedef mtl::compressed2D<real,col_parameters>  col_compressed2D;
//    typedef mtl::matrix::inserter<row_compressed2D>  row_inserter;
//    typedef mtl::matrix::inserter<col_compressed2D>  col_inserter;
//
//    row_compressed2D A( 4, 4 );
//    col_compressed2D B( 4, 4 ), C( 4, 4 );
//    ::blaze::timing::WcTimer timer;
//
//    {
//       row_inserter ins( A );
//
//       ins[0][0] = 1.0;
//       ins[0][2] = 2.0;
//       ins[1][1] = 3.0;
//       ins[1][3] = 4.0;
//       ins[2][0] = 5.0;
//       ins[2][2] = 6.0;
//       ins[3][1] = 7.0;
//       ins[3][3] = 8.0;
//    }
//    {
//       col_inserter ins( B );
//
//       ins[0][0] = 1.0;
//       ins[0][2] = 2.0;
//       ins[1][1] = 3.0;
//       ins[1][3] = 4.0;
//       ins[2][0] = 5.0;
//       ins[2][2] = 6.0;
//       ins[3][1] = 7.0;
//       ins[3][3] = 8.0;
//    }
//
//    std::cout << "\n A =\n" << A << "\n";
//    std::cout << "\n B =\n" << B << "\n";
//
//    C = A * B;
//    std::cout << "\n C =\n" << C << "\n";



   const size_t N( 1000UL );

   typedef ::mtl::tag::row_major  row_major;
   typedef ::mtl::matrix::parameters<row_major>  parameters;
   typedef ::mtl::compressed2D<real,parameters>  compressed2D;
   typedef ::mtl::matrix::inserter<compressed2D>  inserter;

   const size_t NN( N*N );

   compressed2D A( NN, NN );
   ::blaze::timing::WcTimer timer;

   timer.start();
   {
      inserter ins( A );

      for( size_t i=0UL; i<N; ++i ) {
         for( size_t j=0UL; j<N; ++j ) {
            if( i > 0UL   ) ins[i*N+j][(i-1UL)*N+j] = -1.0;  // Top neighbor
            if( j > 0UL   ) ins[i*N+j][i*N+j-1UL  ] = -1.0;  // Left neighbor
            ins[i*N+j][i*N+j] = 4.0;
            if( j < N-1UL ) ins[i*N+j][i*N+j+1UL  ] = -1.0;  // Right neighbor
            if( i < N-1UL ) ins[i*N+j][(i+1UL)*N+j] = -1.0;  // Bottom neighbor
         }
      }
   }
   timer.end();
   std::cerr << "\n Setup of matrix A: runtime = " << timer.last() << "\n\n";
   */
   // END MTL TESTS ///////////////////////////////////////////////////////////////////////////////



   // EIGEN TESTS /////////////////////////////////////////////////////////////////////////////////
   /*
   using ::Eigen::Dynamic;
   using ::Eigen::RowMajor;
   using ::Eigen::ColMajor;

//    ::Eigen::SparseVector<real,RowMajor> a( 3 );
//    //::Eigen::Matrix<real,Dynamic,1> a( 3 );
//    a.insert( 0 ) = 1.0;
//    a.insert( 1 ) = 2.0;
//    a.insert( 2 ) = 1.0;
//    std::cout << "\n a =\n" << a << "\n";
//
//    ::Eigen::SparseVector<real,RowMajor> b( 3 );
//    b.insert( 2 ) = 2.0;
//    std::cout << "\n b =\n" << b << "\n";
//
//    //::Eigen::SparseVector<real,RowMajor> c( 3 );
//    ::Eigen::Matrix<real,Dynamic,1> c( 3 );
//    c[0] = 1.0;
//    c[1] = 1.0;
//    c[2] = 1.0;
//    c = a + b;
//    std::cout << "\n c =\n" << c << "\n";

   Eigen::SparseMatrix<double,RowMajor,int> A( 3, 3 );
   A.insert(0,0) = 1.0;
   A.insert(0,2) = 2.0;
   A.insert(1,1) = 3.0;
   A.insert(2,0) = 4.0;
   A.insert(2,2) = 5.0;

   Eigen::SparseMatrix<double,ColMajor,int> B( 3, 3 );
   B.insert(0,0) = 2.0;
   B.insert(0,2) = 4.0;
   B.insert(1,1) = 6.0;
   B.insert(2,0) = 8.0;
   B.insert(2,2) = 10.0;

   Eigen::SparseMatrix<double,ColMajor,int> C;
   C = A * B;
   std::cerr << "\n C =\n" << C << "\n\n";
   */

   /*
   const size_t N( 250UL );

   using Eigen::Dynamic;
   using Eigen::RowMajor;

   const size_t NN( N*N );

   Eigen::SparseMatrix<double,RowMajor,int> A( NN, NN );
   blaze::timing::WcTimer timer;

   timer.start();
   for( size_t i=0UL; i<N; ++i ) {
      for( size_t j=0UL; j<N; ++j ) {
         if( i > 0UL   ) A.insert(i*N+j,(i-1UL)*N+j) = -1.0;  // Top neighbor
         if( j > 0UL   ) A.insert(i*N+j,i*N+j-1UL  ) = -1.0;  // Left neighbor
         A.insert(i*N+j,i*N+j) = 4.0;
         if( j < N-1UL ) A.insert(i*N+j,i*N+j+1UL  ) = -1.0;  // Right neighbor
         if( i < N-1UL ) A.insert(i*N+j,(i+1UL)*N+j) = -1.0;  // Bottom neighbor
      }
   }
   A.makeCompressed();
   A.finalize();
   timer.end();
   std::cerr << "\n Setup of matrix A: runtime = " << timer.last() << "\n\n";
   */
   // END MTL TESTS ///////////////////////////////////////////////////////////////////////////////



   // TIMING TEST /////////////////////////////////////////////////////////////////////////////////
   /*
   using namespace blaze::timing;

   const size_t reps( 5 );

   const size_t steps  ( 15000 );
   unsigned int counter( 0 );

   //--Test specific variables---------------------------------------------------------------------

   setSeed( 306732 );

   //const size_t M( 5000 );
   const size_t N( 1000000 );
   const size_t F(   10000 );

   DynamicVector<double> a( N ), c( N );
   CompressedVector<double> b( N, F ), d( N, F );

//    CompressedMatrix<double,rowMajor> S1( N, N );
//    CompressedMatrix<double,columnMajor> S2( N, N );

//    for( size_t i=0UL; i<N; ++i ) {
//       for( size_t j=0UL; j<N; ++j ) {
//          A(i,j) = rand<double>();
//          B(i,j) = rand<double>();
//       }
//    }

   for( size_t i=0UL; i<N; ++i ) {
      a[i] = blaze::rand<double>();
      c[i] = blaze::rand<double>();
   }

   for( size_t i=0; i<F; ++i ) {
      b[rand<size_t>(0,N-1)] = rand<double>();
   }

//    for( size_t i=0; i<N; ++i ) {
//       for( size_t j=0; j<F; ++j ) {
//          S1(i,rand<size_t>(0,N-1)) = rand<double>();
//       }
//    }

   d = a * c * b;

   //----------------------------------------------------------------------------------------------

   WcTimer timer;

   for( size_t rep=0; rep<reps; ++rep )
   {
      counter = 0;

      //--Initializations for each repetition------------------------------------------------------



      //-------------------------------------------------------------------------------------------

      timer.start();
      for( size_t step=0; step<steps; ++step )
      {
         //--Performance measurement---------------------------------------------------------------

         d = a * c * b;

         if( d.size() != N ) std::cerr << " Line " << __LINE__ << ": ERROR detected!!!\n";

         //----------------------------------------------------------------------------------------
      }
      timer.end();

      //--Finalization for each repetition---------------------------------------------------------



      //-------------------------------------------------------------------------------------------

      std::cout << " " << rep+1 << ". Run:"
                << "   WC-Time = " << timer.last() << " ,"
                << "   counter = " << counter << "\n";
   }

   std::cout << "\n"
             << " Average WC-Time = " << timer.average() << "\n"
             << " Minimum WC-Time = " << timer.min() << "\n\n";

   //--Finalizations-------------------------------------------------------------------------------



   //----------------------------------------------------------------------------------------------
   */
   // END TIMING TEST /////////////////////////////////////////////////////////////////////////////
}
//*************************************************************************************************
