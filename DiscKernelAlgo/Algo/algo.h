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

#ifndef DISC_ALGO_H
#define DISC_ALGO_H

#include "types.h"
#include "Grid/grid.h"
#include "Model/model.h"


class Algo {
public:
    DiscReturn runDiscAlgo(const Model &model,Grid<bool> *grid) const;
    DiscReturn runRobustAlgo(const Model &model,Grid<bool> *grid) const;
private:
    void initGrid(Grid<bool> *grid) const;
    bool isDiscSafe(const Model &model,const State &x,const Grid<bool> &grid) const;
    bool isRobustSafe(const Model &model,const State &x,const Grid<bool> &grid) const;

    Model model_;
    CarData car_;
    RoadData road_;
};


#endif //DISC_ALGO_H
