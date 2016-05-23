#include <ponos.h>
#include <iostream>

int main(int argc, char* argv[]) {
    ponos::ZGrid<int> grid(8, 8);
    int k = 1;
    for (int y = 0; y < 8; ++y) {
      for (int x = 0; x < 8; ++x) {
        grid(x, y) = k++;
        if(grid(x,y) < 10)
          std::cout << "0";
        std::cout << grid(x,y) << " ";
      }
      std::cout << std::endl;
    }
    return 0;
}
