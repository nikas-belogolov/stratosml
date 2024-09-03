#pragma once

#include <vector>
#include <string>
#include <iostream>
using namespace std;

namespace stratos {
    namespace util {

        /// Type Traits
        template<typename T> constexpr bool IsArithmetic = std::is_arithmetic_v<T>;
        template<bool value> using Requires = std::enable_if_t<value, bool>;


        template<typename T>
        void print_vec(const std::vector<T>& vec) {
            for (auto i : vec) {
                cout << i << " ";
            } cout << endl;
        }

    }
}