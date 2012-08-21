//=================================================================================================
/*!
//  \file Runs.cpp
//  \brief Creates logarithmic scaling benchmarks
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

#include <iomanip>
#include <iostream>




//=================================================================================================
//
//  MAIN FUNCTION
//
//=================================================================================================

//*************************************************************************************************
/*!\brief The main function for the creation of logarithmic scaling benchmarks.
//
// \return 0 in case of a successful execution, 1 in case of errors.
*/
int main()
{
   const size_t start( 1 );
   const size_t end  ( 5000 );
   const float inc   ( 1.1 );

   std::cout << std::right;

   size_t tmp( 0 ), counter( 0 );

   for( size_t i=start; i<=end;) {
      std::cout << "(" << std::setw(8) << i << ")\n";
      ++counter;
      tmp = i * inc;
      if( tmp == i ) ++tmp;
      i = tmp;
   }

   std::cout << "\n Number of runs: " << counter << "\n";
}
//*************************************************************************************************
