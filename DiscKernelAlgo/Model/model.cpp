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

#include "model.h"

StateVector Model::dx(const StateVector &x_vec,const Input &u,double kappa) const
{
    StateVector dx;

    double d = x_vec(0);
    double mu= x_vec(1);
    double v = x_vec(2);

    dx(0) = v*std::sin(mu);
    dx(1) = v*std::tan(u.delta)/car_.L - (v*std::cos(mu))/(1.0-d*kappa)*kappa;
    dx(2) = u.a;

    return dx;
}

State Model::RK4(const State &x,const Input &u,double kappa) const
{
    StateVector x_vec = x.toVector();
    StateVector k1 = dx(x_vec,u,kappa);
    StateVector k2 = dx(x_vec+TS/2.*k1,u,kappa);
    StateVector k3 = dx(x_vec+TS/2.*k2,u,kappa);
    StateVector k4 = dx(x_vec+TS*k3,u,kappa);

    x_vec = x_vec + TS*(k1/6.+k2/3.+k3/3.+k4/6.);

    return {x_vec(0),x_vec(1),x_vec(2)};
}

ModelOutput Model::nextState(const State &x,const Input &u,double kappa) const
{

    return {RK4(x,u,kappa),true};


//    State next_state = x;
//    for(int i = 0;i<5;i++)
//    {
//        next_state = RK4(next_state,u,kappa);
//        if(!roadConstraints(next_state))
//            return {next_state,false};
//    }
//    return {next_state,true};


}

bool Model::roadConstraints(const State &x) const
{
    double tightening = car_.l_car*std::sin(std::fabs(x.mu)) + car_.w_car*std::cos(x.mu);

    return x.d + car_.L / 2.0 * std::sin(x.mu) <= road_.w_road - tightening &&
           x.d + car_.L / 2.0 * std::sin(x.mu) >= -road_.w_road + tightening &&
           x.mu >= -road_.heading_max &&
           x.mu <= road_.heading_max;
}

std::vector<Input> Model::possibleInputs(double v) const
{
    std::vector<Input> inputs;

    double delta_high = std::min(car_.delta_max,std::atan(car_.a_lat_max*car_.L/std::pow(v,2.0)));
    double a_high = car_.a_long_max;

    Eigen::VectorXd a_vec;
    Eigen::VectorXd delta_vec;
    delta_vec.setLinSpaced(NGDELTA,-delta_high,delta_high);
    a_vec.setLinSpaced(NGA,-a_high,a_high);

    for(int i=0;i<NGDELTA;i++)
    {
        for(int j=0;j<NGA;j++)
        {
            if(std::pow(std::pow(v,2.0)*std::tan(delta_vec(i))/car_.L,2.0) + std::pow(a_vec(j),2.0) <= std::pow(car_.a_lat_max,2.0))
                inputs.push_back({a_vec(j),delta_vec(i)});
        }
    }

    return inputs;
}
