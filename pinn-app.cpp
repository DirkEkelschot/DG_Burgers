#include <torch/script.h> // One-stop header.
#include <torch/torch.h>


#include <iostream>
#include <memory>

#include <iostream>
#include <stdio.h>
#include <math.h>
#include "src/Polylib.h"
#include "src/basis_functions.h"
#include "src/basis.h"
#include "src/io.h"
#include <fstream>
#include <cstdio>
#include <stdlib.h>
#include <vector>
#include <map>
#include "src/tinyxml.h"
#include <sstream>
#include "src/basis_poly.h"




struct PINN : torch::nn::Module {
    PINN(int input_size, int hidden_size, int output_size) {
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, hidden_size));
        fc3 = register_module("fc3", torch::nn::Linear(hidden_size, output_size));
    }

    torch::Tensor forward(torch::Tensor x) {
        x = torch::tanh(fc1->forward(x));
        x = torch::tanh(fc2->forward(x));
        x = fc3->forward(x);
        return x;
    }

    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};


torch::Tensor compute_pde_residual(torch::Tensor u, torch::Tensor x, torch::Tensor t, float nu) {
    torch::Tensor u_t = torch::autograd::grad({u}, {t}, {}, true, true)[0];
    torch::Tensor u_x = torch::autograd::grad({u}, {x}, {}, true, true)[0];
    torch::Tensor u_xx = torch::autograd::grad({u_x.sum()}, {x}, {}, true, true)[0];
    
    return u_t + u * u_x - nu * u_xx;
}



torch::Tensor compute_loss(std::shared_ptr<PINN> model,
                          const torch::Tensor& colloc_points,
                          const torch::Tensor& bc_points,
                          float nu) {
    // Boundary condition loss
    auto bc_pred = model->forward(bc_points);
    auto bc_loss = torch::mse_loss(bc_pred, torch::zeros_like(bc_pred));
    
    // PDE residual loss
    auto colloc_pred = model->forward(colloc_points);
    auto pde_res = compute_pde_residual(
        colloc_pred,
        colloc_points.select(1, 0),  // x-coordinates
        colloc_points.select(1, 1),  // t-coordinates
        nu
    );
    auto pde_loss = torch::mse_loss(pde_res, torch::zeros_like(pde_res));
    
    return bc_loss + pde_loss;
}




int main(int argc, char* argv[])
{

    // Initialize model and optimizer
    auto model = std::make_shared<PINN>(2, 20, 1);
    torch::optim::LBFGS optimizer(model->parameters());
    
    // Training data
    const float nu = 0.01f;
    auto colloc_points = torch::rand({1000, 2}, torch::requires_grad());
    auto bc_points = torch::rand({100, 2}, torch::requires_grad());

    // Training loop
    for (int epoch = 0; epoch < 1000; ++epoch) {
        // Lambda closure for LBFGS
        auto closure = [&]() {
            optimizer.zero_grad();
            auto loss = compute_loss(model, colloc_points, bc_points, nu);
            loss.backward();
            return loss;
        };
        
        auto loss = optimizer.step(closure);
        
        if (epoch % 100 == 0) {
            std::cout << "Epoch " << epoch 
                      << ": Loss = " << loss.item<float>() << std::endl;
        }
    }
    
    return 0;
    
    // const int num_epochs = 1000;
    // const float nu = 0.01f;  // Viscosity coefficient
    // auto model = std::make_shared<PINN>(2, 20, 1);
    // torch::optim::LBFGS optimizer(model->parameters());

    // // Sample collocation points
    // auto colloc_points = torch::rand({1000, 2}, torch::requires_grad());
    // auto bc_points = torch::rand({100, 2}, torch::requires_grad());

    // for (int epoch = 0; epoch < num_epochs; ++epoch) 
    // {
    //     torch::Tensor loss = optimizer.step(compute_loss);
    //     if (epoch % 100 == 0) {
    //         std::cout << "Epoch " << epoch << ": Loss = " << loss.item<float>() << std::endl;
    //     }
    // }

    // // Generate test grid
    // auto x_lin = torch::linspace(-1, 1, 100);
    // auto t_lin = torch::linspace(0, 1, 100);
    // auto grid = torch::meshgrid({x_lin, t_lin});
    // auto test_points = torch::stack({grid[0].reshape(-1), grid[1].reshape(-1)}, 1);

    // // Predict solution
    // model->eval();
    // auto solution = model->forward(test_points).reshape({100, 100});

    // // Export for visualization (use Python matplotlib)
    // torch::save(solution, "burgers_solution.pt");


}