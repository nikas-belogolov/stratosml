#pragma once
// #include <stratosml/core/optimizers/optimizers.hpp>
#include <stratosml/core/autodiff/autodiff.hpp>
#include <stratosml/core/data/data.hpp>
#include <stratosml/core/optimizers/optimizers.hpp>
#include <iostream>
#include <armadillo>

// using namespace arma;
using namespace stratos::data;
using namespace stratos;
using namespace stratos::autodiff;
using namespace stratos::losses;
using namespace stratos::activations;
using namespace stratos::optimizers;


namespace stratos {
    
    namespace layers {

        class Layer {

        protected:


            std::shared_ptr<var> add_weight(TensorShape shape) {

                this->weights.push_back(std::make_shared<var>(shape));
                return this->weights.back();
            }

            std::shared_ptr<var> add_weight(std::initializer_list<size_t> shape) {
                return this->add_weight(TensorShape(shape));
            }

        public:

            Activation* activation = new Linear();
            size_t units;
            std::vector<std::shared_ptr<var>> weights;
            std::string name;

            Layer() {}

            Layer(size_t units, std::string name = "") : name(name), units(units) {
                if (units < 1)
                    throw std::invalid_argument("The number of neurons in a layer should be bigger than zero.");
            }

            ~Layer() {
                delete this->activation;
            }

            virtual var forward(ConstantOrVariable<float>& inputs) = 0;

            virtual void build(TensorShape input_shape) {}

        };

    }


}