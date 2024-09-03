#include <stratosml/core/autodiff/autodiff.hpp>

using namespace stratos::autodiff;

namespace stratos {

    namespace activations {

        struct Activation {
            virtual var operator()(const var& input) = 0;
        };

        struct Linear : Activation {
            var operator()(const var& input) override {
                return input;
            }
        };

        struct Sigmoid : Activation {

        };

        struct Relu : Activation {

        };

        struct SoftMax : Activation {

        };


    }

}