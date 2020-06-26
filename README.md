# AdversarialRoadModel
Discriminating kernel algorithm implementation and neural network approximation, as presented in "Safe Motion Planning for Autonomous Driving using an Adversarial Road Model"

## Idea
In our RSS 2020 paper "Safe Motion Planning for Autonomous Driving using an Adversarial Road Model" we show how we can establish a safe predictive path following controller. For this task one big issue is what happens after our prediction horizon since we can plan a trajectory that remains within the constraints until the end of the horizon but depending on how the road evolves, we may not be able to stay within the road indefinitely. For example, if there is a sharp turn directly after the prediction horizon, we may not have enough time to react once we see the turn. 

In this paper, we propose to treat the road after the prediction horizon as an adversarial player, and we establish safe sets, for which we can show that whatever the adversarial road looks like we can follow the road while staying safely inside. 

We formalize this idea by using a kinematic bicycle model in curvilinear coordinates and considering the road curvature as an adversarial player

## Discriminating Kernel

In this setting, the largest safe set is equivalent to the discriminating kernel or the maximum robust control invariant set (depending on the feedback structure of the game). In our paper we focused on the discriminating kernel as it results in a larger set, using the reasonable assumption that the adversarial player plays first, which boils down to the assumption that curvature of the road can be observed.

## Code

In this repo, you can find a C++ implementation that computes the discriminating kernel using a discrete space discriminating kernel algorithm as presented in the paper. But you can also compute the robust control invariant set. Additionally we also included a PyTorch implementation that learns the safe set from the several discrete space discriminating kernels computed for different "maximum curvatures". The resulting network is able to interpolate over a large range of "maximum curvatures" using a small multi-layer perceptron that can be included in an MPC as a terminal constraint.

<img src="https://github.com/alexliniger/AdversarialRoadModel/blob/master/Images/Disc285.png" width="700" />

Visualization of the discriminating kernel, left neural network apporximiation, and right discrete space discriminating kernel, for a test set example where the "maximum curvatures" is 0.0035 1/m.