## Pegasus: Bridging Polynomial and Non-polynomial Evaluations in Homomorphic Encryption

This repo is the implmentation of the paper `Pegasus: Bridging Polynomial and Non-polynomial Evaluations in Homomorphic Encryption`.
Note that this repo should only be used for research and its code are still under heavy developments.

### Requirements
- git 
- c++ compiler that supports at least C++14 standard
- cmake version >= 3.10
- GMP

### Out-of-source Build
1. `mkdir -p build-release & cd build-release`
2. `cmake .. -DSEAL_USE_ZLIB=OFF -DSEAL_USE_MSGSL=OFF -DCMAKE_BUILD_TYPE=Release`
3. `make -j4`
4. check the executables in `build-release/bin/`

### Examples
****

#### Homomorphic Look-up Tables

* codes `examples/lut.cc`
* This demo show the LUTs in Pegasus.

#### Repacking LUT Results in Pegasus
* codes `examples/repacking.cc`
* This demo starts with a CKKS ciphertext that encodes a vector. 
  Then we perform CoeffsToSlots and extraction on the CKKS ciphertext, and gets many LWE ciphertexts.
  After that, we perform LUT on the LWE ciphertexts. 
  Finally, we pack a batch of LUT results (i.e., LWE ciphertexts) into a single CKKS ciphertext.
