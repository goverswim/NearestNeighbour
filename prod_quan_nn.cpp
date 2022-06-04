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
#include <map>

namespace bdap {

    // Constructor, modify if necessary for auxiliary structures
    ProdQuanNN::ProdQuanNN(std::vector<Partition>&& partitions)
        : partitions_(std::move(partitions))
    {}

    void
    ProdQuanNN::initialize_method()
    {
        //std::cout << "Construct auxiliary structures here" << std::endl;
        for (int i = 0; i < (int) this->npartitions(); i++){
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
        /*
        * Compute the nearest neighbours of the examples using PQNN
        * examples: the examples with features that require their NN
        * nneighbors: the number of neighbours for a sample
        * out_index: empty matrix to place the indexes of the neighbours in
        * out_distance: empty matrix to place the distances to the neighbours in
        */
        std::vector<std::vector<float>> clusterDistance;
        for (int i = 0; i < (int) this->npartitions(); i++){
            clusterDistance.push_back( std::vector<float>(this->nclusters(i)));
        }

        for (int i = 0; i < (int) examples.nrows; i++){
            this->compute_nearest_single(examples, nneighbors, out_index, out_distance, i, clusterDistance);
        }


    }


    void ProdQuanNN::compute_nearest_single(const pydata<float>& examples, int nneighbors, pydata<int>& out_index, pydata<float>& out_distance, int index,
    std::vector<std::vector<float>>& clusterDistance) const{
        /*
        * Compute the nearest neighbours of a single sample
        * 
        * examples: examples containing the sample
        * nneighbors: the amount of nearest neighbours required
        * out_index: matrix to place the indices of the neighbours
        * out_distance: matrix to place the distance to the neighbours
        * index: index of the sample in examples
        * clusterDistance: matrix to place the distance of the sample to each cluster
        */

        // Initialize the distance to each cluster
        for (int p = 0; p < (int) this->npartitions(); p++){
            for (int c = 0; c < (int) this->nclusters(p); c++){
                clusterDistance[p][c] = this->distance_to_cluster(examples, index, p, c);
            }
        }

        // KNN
        std::map<float, int> queue;
        float max_distance = 0;
        for (int i=0; i < (int) this->ntrain_examples(); i++){
            float distance = this->distance_to_label(i, clusterDistance);
            if (queue.size() < (size_t) nneighbors){
                queue[distance] = i;
                max_distance = std::max(max_distance, distance);
            } else if (distance < max_distance){
                queue.erase(queue.rbegin()->first);
                queue[distance] = i;
                max_distance = queue.rbegin()->first;
            }
        }
        
        int neighbor = 0;
        
        for ( auto it = queue.begin(); it != queue.end(); ++it ){
            float dist = it->first;
            int neighbor_index = it->second;
            out_index.set_elem(index, neighbor, std::move(neighbor_index));
            out_distance.set_elem(index, neighbor, std::move(dist));
            neighbor++;
        }
    }


    float ProdQuanNN::distance_to_cluster(const pydata<float>& examples, int example_index, size_t partition, size_t cluster) const{
        /*
        * Get the distance of a sample to a given cluster
        * 
        * examples: matrix containing the sample
        * example_index: index of the sample in examples
        * partition: index of the partition of the cluster
        * cluster: index of the cluster (inside a partition)
        * 
        * return float: the distance of the sample to a cluster
        */
        const float* ptr = this->centroid(partition, cluster);
        int begin_feature = this->feat_begin(partition);
        float distance = 0;
        for (int i = begin_feature; i < (int) this->feat_end(partition); i++){
            distance += std::pow(*(ptr + (i - begin_feature)) - examples.get_elem(example_index, i), 2);
        }
        return distance;
    }

    float ProdQuanNN::distance_to_label(size_t label_index, std::vector<std::vector<float>>& clusterDistance) const{
        /*
        * Get the distance between a given test sample and a given label/training sample
        * 
        * label_index: index of the label sample
        * clusterDistance: matrix with the distance of the test sample to each cluster
        * 
        * return float: approximated distance of the test sample to the train sample
        */
        float distance = 0;
        for (int p = 0; p < (int) this->npartitions(); p++){
            
            int cluster = *(this->labels(p)+ label_index);

            distance += clusterDistance[p][cluster];
        }
        return std::sqrt(distance);
    }
} // namespace bdap
