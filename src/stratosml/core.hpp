#pragma once

#include <iostream>
#include <fstream> 
#include <string>
#include <vector>
#include <variant>
#include <unordered_map>
#include <iomanip>
#include <armadillo>
#include <chrono>

#include <stratosml/core/data/data.hpp>
#include <stratosml/core/autodiff/autodiff.hpp>
// #include <stratosml/core/optimizers/optimizers.hpp>
#include <stratosml/core/layers/layers.hpp>
// #include <stratosml/core/activations/activations.hpp>
// #include <stratosml/linear_regression.hpp>

using namespace stratos::layers;

namespace stratos {

    class Model : public Layer {

        

        std::vector<Layer*> layers;

    public:

        Optimizer* optimizer;
        Loss* loss_fn;

        Model() {
            this->optimizer = new GradientDescent(0.001);
            this->loss_fn = new MeanSquaredError();
        }

        ~Model() {
            delete this->optimizer;
            delete this->loss_fn;
        }

        var forward(ConstantOrVariable<float>& inputs) override {
            var output = inputs;

            for (Layer* layer : layers) {
                output = layer->forward(output);
            }

            return output;
        }

        void Add(Layer* layer) {
            this->layers.push_back(layer);
        }

        void Fit(Series& x, Series& y, size_t epochs) {

            constant x_train = x.data;
            constant y_train = x.data;

            this->Fit(x_train, y_train, epochs);

        }

        void Fit(constant& x, const constant& y, size_t epochs) {

            TensorShape input_shape = x->val.shape;

            // Build layers with corresponding input shapes
            for (Layer* layer : layers) {
                layer->build(input_shape);
                input_shape = TensorShape({ layer->units });
            }

            std::vector<std::shared_ptr<var>> parameters;

            for (const Layer* layer : layers) {
                for (const auto& param : layer->weights) {
                    parameters.push_back(param);
                }
            }

            this->optimizer->build(parameters);

            for (int epoch = 1; epoch <= epochs; ++epoch) {
                auto start = std::chrono::high_resolution_clock::now();
                
                // Forward pass
                var output = forward(x);

                // Calculate loss
                var loss = (*loss_fn)(y, output);

                // Backward pass
                loss->derive(1.0);

                // Optimize weights
                this->optimizer->step(parameters);

                // Perform validation here

                auto end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> epoch_time = end - start;

                std::cout << "Epoch " << epoch << "/" << epochs << "\n";
                std::cout << std::fixed << epoch_time.count() << "s - loss: " << std::scientific << loss->val << endl;

                // Update learning rate
                this->optimizer->lr_scheduler->step(epoch);
            }
        }

        var Predict(const Series& x) {
            constant x_pred = x.data;
            return this->Predict(x_pred);
        }

        var Predict(constant& x) {
            var pred = forward(x);
            return pred;
        }

        void Evaluate(Series& x_test, Series& y_test) {
            constant x = x_test.data;
            constant y = y_test.data;
            this->Evaluate(x, y);
        }

        void Evaluate(constant& x_test, constant& y_test) {
            
        }
    };

}
