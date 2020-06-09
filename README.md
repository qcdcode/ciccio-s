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


- Eigen is optional

- C++14 is needed for perfect forwarding, integer sequences and other
  stuff. It could be painfully degradadet to c++11. I need to check
  nvcc support


ASM reporting

When --enable-assembly_report is issued, a .s file is compiled aside a
selection of the .o files, and the file itself is analyzed to check
for bookmarked sequences, which are spit into dedicated files