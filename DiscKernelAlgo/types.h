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

#ifndef DISC_TYPES_H
#define DISC_TYPES_H

#include <Eigen/Dense>

#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

#include <vector>
#include <memory>
#include <string>
#include <algorithm>

#define NX 3
#define NU 2

#define TS 0.2

// training grid
#define NGD 101
#define NGMU 81
#define NGV 135
// test set grid
// #define NGD 201
// #define NGMU 161
// #define NGV 270

#define NGA 9
#define NGDELTA 9

#define NGKAPPA 5

typedef Eigen::Matrix<double,NX,1> StateVector;
typedef Eigen::Matrix<double,NU,1> InputVector;

struct State{
    double d;
    double mu;
    double v;

    StateVector toVector() const{
        StateVector xk;
        xk << d,mu,v;
        return xk;
    }

    void setZero(){
        d = 0;
        mu = 0;
        v = 0;
    };

    State(){
        d = 0;
        mu = 0;
        v = 0;
    }

    State(StateVector x){
        d = x(0);
        mu = x(1);
        v = x(2);
    }
    State(double d_in,double mu_in,double v_in):
    d(d_in),mu(mu_in),v(v_in) {}

};

struct Input{
    double a;
    double delta;
};

struct ModelOutput{
    const State x;
    const bool feasible;
};

struct CarData{
    double l_car;
    double w_car;
    double L;
    double a_lat_max;
    double a_long_max;
    double delta_max;

    CarData()
    {
        using json = nlohmann::json;
        std::ifstream iCar("car.json");
        json jsonCar;
        iCar >> jsonCar;

        l_car = jsonCar["l_car"];
        w_car = jsonCar["w_car"];
        L    = jsonCar["L"];
        a_lat_max = jsonCar["a_lat_max"];
        a_long_max = jsonCar["a_lat_max"];
        delta_max = jsonCar["delta_max"];
    }
};

struct RoadData{
    double w_road;
    double heading_max;
    double velo_max;
    double kappa_max;

    RoadData()
    {
        using json = nlohmann::json;
        std::ifstream iCar("car.json");
        json jsonCar;
        iCar >> jsonCar;

        w_road = jsonCar["w_road"];
        heading_max = jsonCar["heading_max"];
        velo_max = jsonCar["velo_max"];
        kappa_max = jsonCar["kappa_max"];

        velo_max = std::min(std::sqrt((double)jsonCar["a_lat_max"]/kappa_max),velo_max);
    }
};

struct Index{
    int d;
    int mu;
    int v;

    int columnMajorIndex() const{
        return d + NGD*mu + NGD*NGMU*v;
    }

    bool isValid() const {
        if(d<0 || d>=NGD)
            return false;
        else if(mu<0 || mu>=NGMU)
            return false;
        else if(v<0 || v>=NGV)
            return false;
        else
            return true;
    }
};

struct DiscReturn{
    bool converged;
    int n_interation;
};




#endif //DISC_TYPES_H
