#include <stratosml/core/autodiff/node.hpp>

using namespace stratos::autodiff;

namespace stratos {

    namespace losses {

        struct Loss {
            virtual var operator()(const const_or_var& y_true, const const_or_var& y_pred) const = 0;
        };

        struct MeanSquaredError : public Loss {
            var operator()(const const_or_var& y_true, const const_or_var& y_pred) const override {
                return mean(pow(y_pred - y_true, 2));
            }
        };

        struct MeanAbsoluteError : public Loss {
            var operator()(const const_or_var& y_true, const const_or_var& y_pred) const override {
                return mean(abs(y_pred - y_true));
            }
        };

        struct BinaryCrossentropy : public Loss {};
        struct CategoricalCrossentropy : public Loss {};

        // template<typename T>
        // Variable<T> mean_squared_error(const ConstantOrVariable<T>& y_true, const ConstantOrVariable<T>& y_pred) {
        //     return mean(pow(y_pred - y_true, 2));
        // }


    }

}

