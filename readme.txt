Dimuon Properties for Perfect Events
====================================

In the perfect scenario we have mu+/- per event. 

For single tracks the input is;

[charge, station1[/x, y, z/], station3[/x, y, z/], station1[/px, py, pz], station3[/px, py, pz/]]

Target is;

[vertex[/x, y, z, px, py, pz/]]


For dimuons;

For dimuons the input is;

[mu[/+, -/], station1[/x, y, z/], station3[/x, y, z/], station1[/px, py, pz], station3[/px, py, pz/]]

Target is;

[dimuon[/x, y, z, px, py, pz, m, x1, x2/]]


Neural Network Architecture
===========================

* * * * * *     * * * * * * * * * * * * * * *     * * * * * * * * * * * * * * *     * * * * * *
* Input   *     * Classification Layer      *     * Regression layer          *     *         *
* Tensor  * --> * 2 Linear hidden layers    * --> * 3 Linear hidden layers    * --> * Target  *
*         *     * ReLu activation function  *     * ReLu activation function  *     * Tensor  *
* * * * * *     * CrossEntropyLoss          *     * MSELoss                   *     * * * * * *
                * Adam optimizer            *     * Adam optimizer            *
                * * * * * * * * * * * * * * *     * * * * * * * * * * * * * * *

Learning rate: 0.0001
L2 Regularization: 0.00001