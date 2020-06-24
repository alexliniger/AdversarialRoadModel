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

#include "Algo/algo.h"

int main() {

    Model model;
    RoadData road;
    CarData car;

    std::cout << road.velo_max << std::endl;

    Grid<bool> grid(road,car);

    Algo algo;
//    algo.runRobustAlgo(model,&grid);
    algo.runDiscAlgo(model,&grid);

    grid.saveGrid("Disc-Test",road.kappa_max);

    return 0;
}
