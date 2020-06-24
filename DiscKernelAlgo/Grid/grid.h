// Copyright 2020 Alexander Liniger

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
///////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////

#ifndef DISC_GRID_H
#define DISC_GRID_H

#include <types.h>

template <class T>
class Grid{
public:
    Grid(const RoadData road, const CarData &car)
    {
        start_d_ = -road.w_road + car.w_car;
        end_d_   =  road.w_road - car.w_car;
        diff_d_  = (end_d_-start_d_)/(double)(NGD-1);

        start_mu_ = -road.heading_max;
        end_mu_   =  road.heading_max;
        diff_mu_  = (end_mu_-start_mu_)/(double)(NGMU-1);

        start_v_ =  0.0;
        end_v_   =  road.velo_max; //or based on kappa_max
        diff_v_  = (end_v_-start_v_)/(double)(NGV-1);
    };

    ~Grid(){
        std::cout << "deleting grid\n";
    };

    Index projectOntoGrid(const State &x) const
    {
        int dIndex  = (int)round((x.d -start_d_) /diff_d_);
        int muIndex = (int)round((x.mu-start_mu_)/diff_mu_);
        int vIndex  = (int)round((x.v -start_v_) /diff_v_);

        return {dIndex,muIndex,vIndex};
    }

    T getValue(const Index &index) const
    {
        return grid[index.columnMajorIndex()];
    };

    void setValue(const Index &index,T value)
    {
        grid[index.columnMajorIndex()] = value;
    };

    State indexToState(const Index &ind) const
    {
        return {start_d_ +ind.d *diff_d_,
                start_mu_+ind.mu*diff_mu_,
                start_v_ +ind.v *diff_v_};
    }

    void saveGrid(const std::string &name,double kappa) const
    {
        std::string name_file = "Data/" + name + "-" + std::to_string((int)(1./kappa)) + ".bin";
        auto myfile = std::fstream(name_file, std::ios::out | std::ios::binary);
        myfile.write((char*)&grid[0], sizeof(T)*NGD*NGMU*NGV);
        myfile.close();
    }
    int nonZeros() const
    {
        int non_zeros = 0;
        for(int i=0;i < NGD*NGMU*NGV;i++){
            if(grid[i])
                non_zeros++;
        }
        return non_zeros;
    }
private:
    std::unique_ptr<T[]> grid = std::unique_ptr<T[]>(new T[NGD*NGMU*NGV]);

    double diff_d_;
    double diff_mu_;
    double diff_v_;

    double start_d_;
    double start_mu_;
    double start_v_;

    double end_d_;
    double end_mu_;
    double end_v_;
};

#endif //DISC_GRID_H
