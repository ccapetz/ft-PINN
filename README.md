# ft-PINN

# Abstract

Physics-informed neural networks (PINNs) have seen increased utility due to their efficiency and accuracy when modeling complex physics systems, which are often governed by PDEs. To solve more complex PDEs or achieve better expressiveness, larger network architectures are required. Yet, the sensitivity of PINNs often results in ineffective compression. To resolve this problem, we propose a fully-tensorized PINN (ft-PINN) based on the Tensor-Train (TT) decomposition and tensorized training techniques. Additionally, we provide an exploration of PINN architectures to optimize compression ratios whilst maintaining or improving the model’s original expressiveness. To illustrate the benefits of our ft-PINN and advance past PDE solvers, we select and solve a non-linear PDE, the viscous Burgers' equation. Additionally, our ft-PINN accelerates both training and inferencing efficiency, maintains high accuracies, offers up to 98% overall parameter reduction, and significantly outperforms PINNs of similar size.  

# Functionality

Move all contents from FT-PINNs, STD-PINNs, and MISC into the main folder for code functinoality.
