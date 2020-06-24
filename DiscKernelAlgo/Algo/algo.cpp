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

#include "algo.h"

void Algo::initGrid(Grid<bool> *grid) const
{

    for (int i = 0; i < NGD; i++) {
        for (int j = 0; j < NGMU; j++) {

            State x = grid->indexToState({i,j,0});
            double tightening = car_.l_car*std::sin(std::fabs(x.mu)) + car_.w_car*std::cos(x.mu);
            bool safe = x.d + car_.L / 2.0 * std::sin(x.mu) <=  road_.w_road - tightening
                     && x.d + car_.L / 2.0 * std::sin(x.mu) >= -road_.w_road + tightening;

            for (int k=0; k<NGV ; k++)
            {
                grid->setValue({i,j,k},safe);
            }
        }
    }
}

bool Algo::isDiscSafe(const Model &model,const State &x,const Grid<bool> &grid) const
{
    Eigen::VectorXd kappa_vec = Eigen::VectorXd::LinSpaced(NGKAPPA,-road_.kappa_max,road_.kappa_max);

    std::vector<Input> allowed_inputs = model.possibleInputs(x.v);
    bool safe = false;

    for(int i = 0;i<NGKAPPA;i++)
    {
        safe = false;
        for (auto allowed_input : allowed_inputs) {
            ModelOutput next_state = model.nextState(x, allowed_input,kappa_vec(i));
            if(next_state.feasible)
            {
                Index idx = grid.projectOntoGrid(next_state.x);
                if(idx.isValid() && grid.getValue(idx))
                {
                    safe = true;
                    break;
                }
            }
        }
        if(!safe)
            return false;
    }
    return true;
}

bool Algo::isRobustSafe(const Model &model,const State &x,const Grid<bool> &grid) const
{
    Eigen::VectorXd kappa_vec = Eigen::VectorXd::LinSpaced(NGKAPPA,-road_.kappa_max,road_.kappa_max);

    std::vector<Input> allowed_inputs = model.possibleInputs(x.v);
    bool safe = false;
    for (auto allowed_input : allowed_inputs)
    {
        safe = true;
        for(int i = 0;i<NGKAPPA;i++)
        {
            ModelOutput next_state = model.nextState(x, allowed_input,kappa_vec(i));
            if(next_state.feasible)
            {
                Index idx = grid.projectOntoGrid(next_state.x);
                if(!idx.isValid()||!grid.getValue(idx))
                {
                    safe = false;
                    break;
                }
            }
        }
        if(safe)
            return true;
    }
    return false;
}

DiscReturn Algo::runDiscAlgo(const Model &model, Grid<bool> *grid) const
{
    initGrid(grid);
    std::cout << "Initial grid nnz: " << grid->nonZeros() << std::endl;
    int changes = 0;
    for(int it = 0; it < 200; it++)
    {
        std::cout << it << std::endl;
        changes = 0;
        for (int i = 0; i<NGD; i++)
        {
            for (int j = 0; j<NGMU; j++)
            {
                for (int k = 0; k<NGV; k++)
                {
                    if(grid->getValue({i,j,k}))
                    {
                        State x = grid->indexToState({i,j,k});
                        if(!isDiscSafe(model_,x,*grid))
                        {
                            changes++;
                            grid->setValue({i,j,k},false);

                        }
                    }

                }
            }
        }
        std::cout << changes << std::endl;

        std::cout << "non zeros in grid: " << grid->nonZeros() << std::endl;
        if(changes == 0)
        {
            return {true,it+1};
        }
    }
    return {true,0};
}

DiscReturn Algo::runRobustAlgo(const Model &model, Grid<bool> *grid) const
{
    initGrid(grid);

    for(int it = 0; it < 10; it++)
    {
        std::cout << it << std::endl;
        int changes = 0;
        for (int i = 0; i<NGD; i++)
        {
            for (int j = 0; j<NGMU; j++)
            {
                for (int k = 0; k<NGV; k++)
                {
                    if(grid->getValue({i,j,k}))
                    {
                        State x = grid->indexToState({i,j,k});
                        if(!isRobustSafe(model_,x,*grid))
                        {
                            changes++;
                            grid->setValue({i,j,k},false);
                        }
                    }

                }
            }
        }
        std::cout << changes << std::endl;
        if(changes == 0)
        {
            return {true,it+1};
        }
    }
    return {true,0};
}