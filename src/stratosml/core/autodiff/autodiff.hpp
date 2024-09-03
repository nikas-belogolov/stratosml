#pragma once

#include <stratosml/core/autodiff/tensor.hpp>
#include <stratosml/core/autodiff/node.hpp>
#include <stratosml/core/autodiff/ops.hpp>

#include <stratosml/core/autodiff/nn/losses.hpp>
#include <stratosml/core/autodiff/nn/activations.hpp>

namespace stratos {

    namespace autodiff {

     /*
        template<typename... Args>
        struct Wrt {
            std::tuple<Args...> args;
        };

        // Template function to create ArgsStruct from any input
        template<typename T>
        Wrt<T> wrt(const T& var) {
            // Create ArgsStruct with a tuple containing the variable
            return Wrt<T>{ std::make_tuple(var) };
        }

        template<typename... Args>
        Wrt<Args...> wrt(const Args&... vars) {
            // Create ArgsStruct with a tuple containing multiple variables
            return Wrt<Args...>{ std::make_tuple(vars...) };
        }

        template<size_t i>
        struct Index
        {
            constexpr static size_t index = i;
            constexpr operator size_t() const { return index; }
            constexpr operator size_t() { return index; }
        };

        template<std::size_t i = 0, std::size_t N, typename Func>
        constexpr auto For(Func&& func) {
            if constexpr (i < N) {
                func(Index<i>{});
                For<i + 1, N>(std::forward<Func>(func));
            }
        }

        template<std::size_t N, typename Func>
        constexpr auto For(Func&& func) {
            For<0, N>(std::forward<Func>(func));
        }

        template<typename Tuple, std::size_t... Is>
        auto copyTuple(const Tuple& t, std::index_sequence<Is...>) {
            return std::make_tuple((std::get<Is>(t).expr->val)...);
        }

        template<typename... Args>
        auto copyTuple(const std::tuple<Args...>& t) {
            return copyTuple(t, std::index_sequence_for<Args...>{});
        }

        template<typename T, typename... Args>
        auto gradients(T y, Wrt<Args...> wrt) {

            constexpr std::size_t N = std::tuple_size_v<decltype(wrt.args)>;

            // Create a tuple of gradient variables with the same type and size as the original variables
            // This ensures that each gradient is compatible with its corresponding variable for proper differentiation
            auto grads = copyTuple(wrt.args);

            // For<N>([&](auto i) constexpr {
            //     std::get<i>(wrt.args).expr->bind_value(&std::get<i>(grads));
            //     std::get<i>(grads) = {0};
            // });


            using GradType = decltype(y->val);

            GradType grad;

            if constexpr (std::is_arithmetic<GradType>::value) {
                grad = T(1);
            } else if constexpr (arma::is_arma_type<GradType>::value) {
                grad = GradType(arma::size(y->val), arma::fill::ones);
            }
            
            y->derive(grad);

            // For<N>([&](auto i) constexpr {
            //     std::get<i>(wrt.args).expr->bind_value(nullptr);
            // });

            return grads;
        }

        */


    }

}