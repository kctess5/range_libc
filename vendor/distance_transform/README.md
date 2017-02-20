C++ Distance Transform
======================

This software is a C++11 implementation of the algorithm described in:

>** * Distance Transforms of Sampled Functions * **  
>Pedro F. Felzenszwalb, Daniel P. Huttenlocher  
>Theory of Computing, Vol. 8, No. 19, September 2012

See their [project site](http://cs.brown.edu/~pff/dt/).  

The paper describes a linear-time algorithm for solving a class of minimization
problems involving a cost function with both local and spatial terms.
Such problems can be viewed as a generalization of classical distance transforms
of binary images, where the binary image is replaced by an arbitrary sampled
function.
Consequently, it can be used as a simple, fast method for computing the
Euclidean distance transform of a binary image.

###### Note ######
This software takes some cues from both their implementation and the one by Sofien Bouaziz and Andrea Tagliasacchi ([check here](https://github.com/ataiya/dtform)).

## Features: ##
* ***N*-Dimensional:** The implementation by Felzenszwalb and Huttenlocher
    works only with *2*-D images. Here I provide an implementation that works
    with arrays of any dimension *N*.
* **Low memory usage:** By using
    [dope vectors](https://github.com/giorgiomarcias/DopeVector), there is not need
    of copying array data around during the computation.
* **Index of nearest element:** Optionally get the index (as the linear
    distance from the beginning of a mono-dimensional array - see the example)
    of the nearest element to the one at any position in the array. Not just its
    distance (taken from the implementation by Sofien Bouaziz and Andrea
    Tagliasacchi).
* **Parallel execution by giving the number of threads to spread.

### License ###
This software is subject to the [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0.html) License.
