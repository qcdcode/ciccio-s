# ciccio-s
Francesco's version of Kokkos

To bootstrap:

```
bash config/bootstrap
```

To compile:

```
mkdir build
cd build
../configure \
	     "CXX=g++" \
	     "CXXFLAGS=-O3 -Wall -std=c++14 -mavx #or avx512f" \
	     --with-eigen=(dir where Eigen/Dense is located) \
	     --with-simd-inst-set=(avx [default] or avx512) \
	     --disable-openmp #so far not used \
	     --enable-assembly_report #To output parsed .s file for each highlighted region
make

```

To test:
```
bin/main
```


Eigen is optional
C++14 is needed for perfect forwarding, integer sequences and other stuff. We could painfully degrade to c++11