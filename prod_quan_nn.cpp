/*
 * author: Laurens Devos
 * Copyright BDAP team, DO NOT REDISTRIBUTE
 *
 *******************************************************************************
 *                                                                             *
 *                     DO NOT CHANGE SIGNATURES OF METHODS!                    *
 *             DO NOT CHANGE METHODS IMPLEMENTED IN THIS HEADER!               *
 *     Sections which require modifications indicated with 'TODO' comments     *
 *                                                                             *
 *******************************************************************************
 */

#include "prod_quan_nn.hpp"
#include <limits>
#include <chrono>
#include <cmath>
#include <unordered_map>

namespace bdap {

    // Constructor, modify if necessary for auxiliary structures
    ProdQuanNN::ProdQuanNN(std::vector<Partition>&& partitions)
        : partitions_(std::move(partitions))
    {}

    void
    ProdQuanNN::initialize_method()
    {
        //std::cout << "Construct auxiliary structures here" << std::endl;
        for (int i = 0; i < this->npartitions(); i++){
            this->clusterDistance.push_back( std::vector<float>(this->nclusters(i)));
        }
    }

    void
    ProdQuanNN::compute_nearest_neighbors(
                const pydata<float>& examples,
                int nneighbors,
                pydata<int>& out_index,
                pydata<float>& out_distance) const
    {
        std::vector<std::vector<float>> clusterDistance;
        for (int i = 0; i < this->npartitions(); i++){
            clusterDistance.push_back( std::vector<float>(this->nclusters(i)));
        }
        std::cout << "test";
        for (int i = 0; i < examples.nrows; i++){
            this->compute_nearest_single(examples, nneighbors, out_index, out_distance, i, clusterDistance);
        }
        
        //std::cout << "Compute the nearest neighbors for the "
        //    << examples.nrows
        //    << " given examples." << std::endl

        //    << "The examples are given in C-style row-major order, that is," << std::endl
        //    << "the values of a row are consecutive in memory." << std::endl

        //    << "The 5th example can be fetched as follows:" << std::endl;

        //float const *ptr = examples.ptr(5, 0);
        //std::cout << '[';
        //for (size_t i = 0; i < examples.ncols; ++i) {
        //    if (i>0) std::cout << ",";
        //    if (i>0 && i%5==0) std::cout << std::endl << ' ';
        //    printf("%11f", ptr[i]);
        //}
        //std::cout << " ]" << std::endl;



    }
    void ProdQuanNN::compute_nearest_single(const pydata<float>& examples, int nneighbors, pydata<int>& out_index, pydata<float>& out_distance, int index,
    std::vector<std::vector<float>>& clusterDistance) const{
        const float* ptr = this->centroid(0, 0);
        for (int p = 0; p < this->npartitions(); p++){
            for (int c = 0; c < this->nclusters(p); c++){
                
            }
        }
        clusterDistance[0][0] = examples.get_elem(index, 0) + *(this->centroid(0, 0));
        std::cout << clusterDistance[0][0];
    }
} // namespace bdap
