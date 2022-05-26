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
        //std::cout << "\n" << this->ntrain_examples() << " " << this->npartitions();
        
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
        const float* ptr = this->centroid(partition, cluster);
        int begin_feature = this->feat_begin(partition);
        float distance = 0;
        for (int i = begin_feature; i < (int) this->feat_end(partition); i++){
            distance += std::pow(*(ptr + (i - begin_feature)) - examples.get_elem(example_index, i), 2);
        }
        distance = std::sqrt(distance);
        return distance;
    }

    float ProdQuanNN::distance_to_label(size_t label_index, std::vector<std::vector<float>>& clusterDistance) const{
        float distance = 0;
        for (int p = 0; p < (int) this->npartitions(); p++){
            
            int cluster = *(this->labels(p)+ label_index);

            distance += clusterDistance[p][cluster];
        }
        return distance;
    }
} // namespace bdap
