#include <iomanip>
#include <iostream>

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

