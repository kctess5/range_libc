C++ Dope Vector
===============

This software is a C++11 header-only implementation of [dope vectors](https://en.wikipedia.org/wiki/Dope_vector).

A *dope vector* is a data structure holding information about a data object array, which is typically stored in memory sequentially.
Especially, it provides a layout interface, abstracting the effective underlying array memory sequential layout.
Such interface can be changed, meaning that an array can be seen as an array, a 2D NxM matrix, a 2D MxN matrix, or in general a multi-dimensional matrix of any sizes, as long as the total size does not exceed the original array bounds.

A dope vector, hence, is a proxy between some data and users, allowing fast operations, such as extracting windows and slices or making permutations, with ease.
Suppose you have a function that prints the elements of a matrix:

    void printMatrix(Matrix m);

The suppose you want to print only a subset of the matrix, like a window or a slice, or even you want to print the whole matrix but in a different order, i.e. a permutation.
You have two possibilities: you could modify the function above to pass in the index bounds or the indexes for the permutation, OR you could make new matrix copying its values in the right cells.
With a dope vector, and using it as a general matrix representation, it would be sufficient to pass the result of the window extraction method directly in to the print function:

    using namespace dope;
    template < class T, SizeType D > void print(DopeVector<T, D> m);
    ...
    using Matrix = DopeVector<float,2>;
    Matrix m;
    ...
    print(m.window({0, 1}, {3, 4}));

The extraction of a window, a slice, or even a permutation are all fast operations performed in nearly constant time (precisely they do not depend on the sizes of the matrixes but only on their dimensionality).

The library provides a `Index<D>` class for fast creation and processing of index arrays.
They are used to pass in the sizes and offsets at construction/resize time, or the indexes for accessing windows/slices/permutations.
They might also be used in arithmetic expressions which are lazy-evaluated only when the final value of the expressions are effectively needed.

Moreover, the library provides a `Grid<T, D>` class that shows how to derive from a DopeVector and make `D`-dimensional matrixes with automatic memory management.
Since `DopeVector` is only a wrapper on top of a memory address, and every operation performed on any other object created from a `DopeVector` actually affects the original memory, you must provide and manage the memory allocation.
So `Grid` helps to this.

The library is header-only, just include the path `<path-to-where-downloaded-to>/include` in your project and in your code add

    #include <DopeVector/DopeVector.hpp>
    //or directly:
    //#include <DopeVector/Grid.hpp>

If you use [CMAKE](https://cmake.org), you can import this project using `add_subdirectory`:

    add_subdirectory("<path-to-where-downloaded-to>" "${CMAKE_CURRENT_BINARY_DIR}/DopeVector")
    ...
    target_link_libraries(${PROJECT_NAME} INTERFACE DopeVector)

Note that, due to cmake's design of not exporting file properties of interface libraries (as of version 3.6.2), if you want to have DopeVector's header files in your IDE project tree just add the following:

    option(ATTACH_SOURCES "When generating an IDE project, add DopeVector header files to project sources." ON)
    add_subdirectory("<path-to-where-downloaded-to>" "${CMAKE_CURRENT_BINARY_DIR}/DopeVector")
    include("<path-to-where-downloaded-to>/sources_properties.cmake")
    ...
    target_link_libraries(${PROJECT_NAME} INTERFACE DopeVector)
    set_dope_vector_source_files_properties()


It is also possible to use [Eigen](http://eigen.tuxfamily.org) vectors into expressions involving DopeVector's own Index's.
Just include the path where Eigen is located and define `DOPE_USE_EIGEN_INDEX` in your code.
If you use cmake to import DopeVector, it is possible to set the option

    option(WITH_EIGEN "Build DopeVector with Index<Dimension> as Eigen Matrix (if present) or not." OFF)

or compile with `cmake -DUSE_EIGEN=ON <path-to-where-downloaded-to>`.
This allows you to write things such as:

    Index2 a(2, 3), y;
    Eigen::Vector2i e(4, 5);

    y = a + Index2::Ones() + e;
    


### License ###
This software is subject to the [Apache 2.0](http://www.apache.org/licenses/LICENSE-2.0.html) License.
