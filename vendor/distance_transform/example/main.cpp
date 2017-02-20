// Copyright (c) 2016 Giorgio Marcias
//
// This file is part of distance_transform, a C++11 implementation of the
// algorithm in "Distance Transforms of Sampled Functions"
// Pedro F. Felzenszwalb, Daniel P. Huttenlocher
// Theory of Computing, Vol. 8, No. 19, September 2012
//
// This source code is subject to Apache 2.0 License.
//
// Author: Giorgio Marcias
// email: marcias.giorgio@gmail.com

#include <iostream>
#include <iomanip>
#include <vector>
#include <chrono>

#include <distance_transform/distance_transform.hpp>
using namespace dt;
using namespace dope;

int main(int argc, char *argv[])
{
    Index2 size({15, 15});
    Grid<float, 2> f(size);
    Grid<SizeType, 2> indices(size);
    for (SizeType i = 0; i < size[0]; ++i)
        for (SizeType j = 0; j < size[1]; ++j) {
            if (i == j && i*j < size[0]*size[1] / 2)
                f[i][j] = 0.0f;
            else
                f[i][j] = std::numeric_limits<float>::max();
        }

	// Note: this is necessary at least at the first distance transform execution
	// and every time a reset is desired; it is not, instead, when updating
    DistanceTransform::initializeIndices(indices);

    std::cout << "indices:" << std::endl;
    for (SizeType i = 0; i < size[0]; ++i) {
        for (SizeType j = 0; j < size[1]; ++j)
            std::cout << std::setw(7) << indices[i][j] << ' ';
        std::cout << std::endl;
    }

    std::cout << std::endl << "Window [2:6;3:8]:" << std::endl;
    Index2 winStart({2, 3});
    Index2 winSize({5, 6});
    DopeVector<SizeType, 2> indicesWin = indices.window(winStart, winSize);
    for (SizeType i = 0; i < indicesWin.sizeAt(0); ++i) {
        for (SizeType j = 0; j < indicesWin[i].sizeAt(0); ++j)
            std::cout << std::setw(7) << indicesWin[i][j] << ' ';
        std::cout << std::endl;
    }

    std::cout << std::endl << "Slice 2 at dimension 0:" << std::endl;
    DopeVector<SizeType, 1> sl = indices.slice(0, 2);
    for (SizeType j = 0; j < sl.sizeAt(0); ++j)
        std::cout << std::setw(7) << sl[j] << ' ';
    std::cout << std::endl << "Slice 2 at dimension 1:" << std::endl;
    indices.slice(1, 2, sl);
    for (std::size_t j = 0; j < sl.sizeAt(0); ++j)
        std::cout << std::setw(7) << sl[j] << ' ';
    std::cout << std::endl;

    std::cout << std::endl << "Window [4:9] of slide 2 at dimension 1:" << std::endl;
    sl = sl.window(4, 6);
    for (SizeType j = 0; j < sl.sizeAt(0); ++j)
        std::cout << std::setw(7) << sl[j] << ' ';
    std::cout << std::endl << std::endl;

    std::cout << "f:" << std::endl;
    for (SizeType i = 0; i < size[0]; ++i) {
        for (SizeType j = 0; j < size[1]; ++j)
            std::cout << std::setw(4) << std::setprecision(1) << std::scientific << f[i][j] << ' ';
        std::cout << std::endl;
    }

    /* Full parameter list:     
     *  matrix f,           matrix D,           bool,                                   nThreads
     *  initial distances,  final distances,    keep square distances (true) or not,    number of threads (> 1 parallel, <= 1 sequential)
     * or:                     
     *  matrix f,           matrix D,           matrix I,               bool,                                   nThreads
     *  initial distances,  final distances,    indices of nearests,    keep square distances (true) or not,    number of threads (> 1 parallel, <= 1 sequential)
     * NB:
     *  - by default, squared distances are not kept, but square roots are computed
     *  - by default, the number of threads is automatically set given the hardware cores
     */

    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    DistanceTransform::distanceTransformL2(f, f, indices, false, 1);
    std::cout << std::endl << "2D distance function computed in: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count() << " ns." << std::endl;

    std::cout << std::endl << "D (squared):" << std::endl;
    for (SizeType i = 0; i < size[0]; ++i) {
        for (SizeType j = 0; j < size[1]; ++j)
            std::cout << std::setw(4) << std::setprecision(1) << std::fixed << f[i][j] << ' ';
        std::cout << std::endl;
    }

    std::cout << std::endl << "indices:" << std::endl;
    for (SizeType i = 0; i < size[0]; ++i) {
        for (SizeType j = 0; j < size[1]; ++j)
            std::cout << std::setw(7) << indices[i][j] << ' ';
        std::cout << std::endl;
    }


    for (SizeType i = 0; i < size[0]; ++i)
        for (SizeType j = 0; j < size[1]; ++j)
            f[i][j] = std::numeric_limits<float>::max();
    DopeVector<float, 2> fWin = f.window(winStart, winSize);
    for (SizeType i = 0; i < fWin.sizeAt(0); ++i)
        for (SizeType j = 0; j < fWin[i].sizeAt(0); ++j)
            if (i == 0 || i == fWin.sizeAt(0)-1 || j == 0 || j == fWin[i].sizeAt(0)-1)
                fWin[i][j] = 0.0f;
            else
                fWin[i][j] = std::numeric_limits<float>::max();
    DistanceTransform::initializeIndices(indices);

    std::cout << std::endl << "f reset:" << std::endl;
    for (SizeType i = 0; i < size[0]; ++i) {
        for (SizeType j = 0; j < size[1]; ++j)
            std::cout << std::setw(4) << std::setprecision(1) << std::scientific << f[i][j] << ' ';
        std::cout << std::endl;
    }

    start = std::chrono::steady_clock::now();
    DistanceTransform::distanceTransformL2(fWin, fWin, indicesWin, true, 1);
    std::cout << std::endl << "2D distance function computed on the window in: " << std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count() << " ns." << std::endl;

    std::cout << std::endl << "D (squared):" << std::endl;
    for (SizeType i = 0; i < size[0]; ++i) {
        for (SizeType j = 0; j < size[1]; ++j)
            std::cout << std::setw(4) << std::setprecision(1) << std::scientific << f[i][j] << ' ';
        std::cout << std::endl;
    }

    std::cout << std::endl << "indices:" << std::endl;
    for (SizeType i = 0; i < size[0]; ++i) {
        for (SizeType j = 0; j < size[1]; ++j)
            std::cout << std::setw(7) << indices[i][j] << ' ';
        std::cout << std::endl;
    }



    // 2D
    size = {320, 240};
    Grid<float, 2> f2D(size);
    for (SizeType i = 0; i < size[0]; ++i)
        for (SizeType j = 0; j < size[1]; ++j)
            f2D[i][j] = std::numeric_limits<float>::max();
    f2D[0][0] = 0.0f;
    start = std::chrono::steady_clock::now();
    DistanceTransform::distanceTransformL2(f2D, f2D, false, 1);
    std::cout << std::endl << size[0] << 'x' << size[1] << " distance function computed in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() << " ms." << std::endl;

    // 3D
    Index3 size3D = {400, 320, 240};
    Grid<float, 3> f3D(size3D);
    for (SizeType i = 0; i < size3D[0]; ++i)
        for (SizeType j = 0; j < size3D[1]; ++j)
            for (std::size_t k = 0; k < size3D[2]; ++k)
                f3D[i][j][k] = std::numeric_limits<float>::max();
    f3D[0][0][0] = 0.0f;
    start = std::chrono::steady_clock::now();
    DistanceTransform::distanceTransformL2(f3D, f3D, false, 1);
    std::cout << std::endl << size3D[0] << 'x' << size3D[1] << 'x' << size3D[2] << " distance function computed in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() << " ms." << std::endl;

    // 3D parallel
    for (SizeType i = 0; i < size3D[0]; ++i)
        for (SizeType j = 0; j < size3D[1]; ++j)
            for (SizeType k = 0; k < size3D[2]; ++k)
                f3D[i][j][k] = std::numeric_limits<float>::max();
    f3D[0][0][0] = 0.0f;
    start = std::chrono::steady_clock::now();
    DistanceTransform::distanceTransformL2(f3D, f3D, false, std::thread::hardware_concurrency());
    std::cout << std::endl << size3D[0] << 'x' << size3D[1] << 'x' << size3D[2] << " distance function (concurrently) computed in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() << " ms. (with " << std::thread::hardware_concurrency() << " threads)." << std::endl;

    // 2D big
    size = {10000, 10000};
    Grid<float, 2> f2DBig(size);
    for (SizeType i = 0; i < size[0]; ++i)
        for (SizeType j = 0; j < size[1]; ++j)
            f2DBig[i][j] = std::numeric_limits<float>::max();
    f2DBig[0][0] = 0.0f;
    start = std::chrono::steady_clock::now();
    DistanceTransform::distanceTransformL2(f2DBig, f2DBig, false, 1);
    std::cout << std::endl << size[0] << 'x' << size[1] << " distance function computed in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() << " ms." << std::endl;

    // 2D big parallel
    for (SizeType i = 0; i < size[0]; ++i)
        for (SizeType j = 0; j < size[1]; ++j)
            f2DBig[i][j] = std::numeric_limits<float>::max();
    f2DBig[0][0] = 0.0f;
    start = std::chrono::steady_clock::now();
    DistanceTransform::distanceTransformL2(f2DBig, f2DBig, false, std::thread::hardware_concurrency());
    std::cout << std::endl << size[0] << 'x' << size[1] << " distance function (concurrently) computed in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() << " ms. (with " << std::thread::hardware_concurrency() << " threads)." << std::endl;

    // 6D
    Index<6> size6D = {5, 5, 5, 5, 5, 5};
    Grid<float, 6> f6D(size6D);
    for (SizeType i = 0; i < size6D[0]; ++i)
        for (SizeType j = 0; j < size6D[1]; ++j)
            for (SizeType k = 0; k < size6D[2]; ++k)
                for (SizeType l = 0; l < size6D[3]; ++l)
                    for (SizeType m = 0; m < size6D[4]; ++m)
                        for (SizeType n = 0; n < size6D[5]; ++n)
                            f6D[i][j][k][l][m][n] = std::numeric_limits<float>::max();
    f6D[0][0][0][0][0][0] = 0.0f;
    start = std::chrono::steady_clock::now();
    DistanceTransform::distanceTransformL2(f6D, f6D, false, 1);
    std::cout << std::endl << size6D[0] << 'x' << size6D[1] << 'x' << size6D[2] << 'x' << size6D[3] << 'x' << size6D[4] << 'x' << size6D[5] << " distance function computed in: " << std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - start).count() << " ms." << std::endl;

    std::cout << std::endl;

    return 0;
}
