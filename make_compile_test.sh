make
g++ -std=c++20 -I./include -L./lib -fcompare-debug-second src/stratosml/tests/main.cpp -o tests -larmadillo -lblas -llapack -lstratosml
./tests