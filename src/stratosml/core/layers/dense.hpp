#pragma once

// #include <armadillo>
// #include <stratosml/core/layers/layer.hpp>
// #include <stratosml/core/optimizers/optimizers.hpp>

// using namespace arma;
// using namespace stratos;
// using namespace stratos::optimizers;

namespace stratos {

    namespace layers {

        class Dense : public Layer {
            
            std::shared_ptr<var> biases;
            std::shared_ptr<var> kernel;

        public:

            Dense(size_t units = 1, std::string name = "") : Layer(units, name) {
                this->biases = this->add_weight({ units });
            }

            void build(TensorShape input_shape) {
                this->kernel = this->add_weight({ input_shape[input_shape.rank() - 1], units });
            }

            var forward(ConstantOrVariable<float>& inputs) override {
                var z = (inputs * (*this->kernel)) + (*this->biases);

                return (*this->activation)(z);
            }

        };

    }
}