## Pegasus: Bridging Polynomial and Non-polynomial Evaluations in Homomorphic Encryption

This repo is the implmentation of the paper [Pegasus: Bridging Polynomial and Non-polynomial Evaluations in Homomorphic Encryption](https://eprint.iacr.org/2020/1606).
Note that this repo should only be used for research and its code are still under heavy developments.

**Additional Note**

* The reported performance of the F_mod in our paper used [Han et al.'s method](https://eprint.iacr.org/2019/688.pdf) for the homomorphic Sin function. In this repo, we update the F_mod implementation to [Bossuat et al.'s method](https://eprint.iacr.org/2020/1203.pdf), using the examples from the [Lattigo](https://github.com/ldsec/lattigo) repo.

* This repo provides the implementation of _binary_ [LWE secret key](https://github.com/Alibaba-Gemini-Lab/OpenPEGASUS/blob/master/pegasus/lwe.cc#L59).
  For the ternary secret key, we can use the technique of [this paper](https://eprint.iacr.org/2020/086.pdf) at the cost of doubling the LUT time.

**Additional Note (Update 2021-12-14)** 
* PEGASUS can evaluate any point-wise function `f(x)` defined over the interval `x \in ZZ \caps [-N/2, N/2)`. This interval can be extended to `ZZ \caps [-N, N)` when the target function is negacyclic, i.e., `f(x+N) = f(x)`. As we have pointed in our paper, we assume the CKKS ciphertext should encrypt message in the range `[-q/4, q/4)` so that it can be discreted to the proper range `[-N/2, N/2)`.
* To obtain a full domain look-up-table without the requirements of negacyclic functions, please refer to the paper [FDFB: Full Domain Functional Bootstrapping Towards Practical Fully Homomorphic Encryption](https://eprint.iacr.org/2021/1135.pdf).

## Requirements

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
