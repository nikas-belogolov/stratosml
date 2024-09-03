#pragma once

#include <initializer_list>
#include <armadillo>
#include <stratosml/util.hpp>

/*
 *
 * TENSOR - Wrapper around arma matrices
 * 
 */

using namespace stratos::util;

namespace stratos {

    namespace autodiff {

        template<typename T> using mat = arma::Mat<T>;
        
        struct TensorShape {
        
        public:
            std::vector<size_t> dims;

            TensorShape() {}
            TensorShape(const TensorShape& other) : dims(other.dims) {}

            TensorShape(std::initializer_list<size_t> dims) : dims(dims) {}

            size_t rank() {
                return dims.size();
            }

            const size_t operator[](size_t dim) const {
                return dims[dim];
            }

        };


        // Tensors - Numpy like arrays, immutable
        template<typename T>
        struct Tensor {

            // TensorShape shape;

            TensorShape shape;


            Tensor() {}

            Tensor(const TensorShape& shape) {
                value = mat<T>(shape[0], shape[1] || 1);
                this->shape = shape;
            }

            Tensor(const Tensor<T>& tensor): value(tensor.value), shape(tensor.shape) {}

            template<typename FillForm>
            Tensor(const arma::SizeMat& size, const arma::fill::fill_class<FillForm> fill_form) {
                shape = { size.n_rows, size.n_cols };
                value = arma::Mat<T>(size, fill_form);
            }

            template<typename FillForm>
            Tensor(const Tensor<T>& tensor, const arma::fill::fill_class<FillForm> fill_form) {
                shape = tensor.shape;
                value = arma::Mat<T>(arma::size(tensor.value), fill_form);
            }

            Tensor(T scalar): value(arma::Mat<T>(1, 1, arma::fill::value(scalar))), shape{} {}

            Tensor(const std::initializer_list<T>& vector) : value(arma::Mat<T>(arma::Col<T>(vector))), shape{vector.size()} {}
            
            Tensor(const std::initializer_list<std::initializer_list<T>>& matrix) {
                value = arma::Mat<T>(matrix);
                shape = {value.n_rows, value.n_cols};
            }

            Tensor(arma::Mat<T> matrix): value(matrix), shape{matrix.n_rows, matrix.n_cols} {}

            ~Tensor() {
                // vector.~Col();
            }

            Tensor<T>& operator=(const Tensor<T>& other) {
                this->shape = other.shape;
                this->value = other.value;
                return *this;
            }

            bool is_scalar() const {
                return value.n_cols == 1 && value.n_rows == 1;
            }

            bool is_vector() const {
                return !this->is_scalar() && (value.n_cols == 1 || value.n_rows == 1);
            }

            bool is_col_vector() const {
                return this->is_vector() && value.n_cols == 1;
            }

            bool is_row_vector() const {
                return this->is_vector() && value.n_rows == 1;
            }

            void reshape(arma::SizeMat size) {
                value.reshape(size);
            }

            /// ------------
            /// Arma Methods
            /// ------------

            Tensor<T> t() {
                return Tensor<T>(this->value.t());
            }

            T& operator()(size_t row, size_t col) {
                return this->value(row, col);
            }

            const T& operator()(size_t row, size_t col) const {
                return value(row, col);
            }

            T& operator()(size_t row) {
                return value(row);
            }

            const T& operator()(size_t row) const {
                return value(row);
            }

            operator const arma::Mat<T>&() const {
                return this->value;
            }

            T min() {
                return this->value.min();
            }

            T max() {
                return this->value.max();
            }

            operator T&() {
                return value(0,0);
            }

            /// Tensor assignment operators
            Tensor<T>& operator+=(const Tensor<T>& other) { this->value += this->broadcast(other); return *this; }
            Tensor<T>& operator-=(const Tensor<T>& other) { this->value -= this->broadcast(other); return *this; }
            Tensor<T>& operator*=(const Tensor<T>& other) { this->value *= this->broadcast(other); return *this; }
            Tensor<T>& operator/=(const Tensor<T>& other) { this->value /= this->broadcast(other); return *this; }

            friend std::ostream& operator<<(std::ostream& s, const Tensor<T>& t) {
                s << t.value;
                return s;
            }

            arma::Mat<T> value;
            
        private:

            // Broadcast other tensor to match this tensor shape
            // Used in element-wise assignment operators (+=, -=, /=, %=)
            arma::Mat<T> broadcast(const Tensor<T>& t) {
                arma::Mat<T> result = t.value;

                // cout << "M: " << t << endl;
                // cout << "this: " << this->value << endl;

                // cout << t << endl;

                if (this->is_scalar()) {
                    result = arma::Mat<T>(1,1);
                    result(0,0) = arma::accu(t.value);
                    return result;
                }

                if (t.is_scalar()) {
                    return arma::repmat(result, this->value.n_rows, this->value.n_cols);
                }

                if (this->is_row_vector() && this->value.n_cols == t.value.n_rows) {
                    return arma::sum(t.value, 1).t();
                }
                
                return t;

                // throw std::invalid_argument("Incompatible tensor shapes.");
            }
        };

        struct Initializer {

            virtual Tensor<float> operator()() = 0;

        };

        struct Zeros : Initializer {

            Tensor<float> operator()(TensorShape shape) {
                Tensor<float> t = Tensor<float>(shape);
                return t;
            }

        };

        template<typename T>
        std::pair<arma::Mat<T>, arma::Mat<T>> element_wise_broadcast(const Tensor<T>& t1, const Tensor<T>& t2) {

            mat<T> t1_broadcasted = t1.value;
            mat<T> t2_broadcasted = t2.value;

            if (t1.is_scalar()) {
                t1_broadcasted = arma::repmat(t1_broadcasted, t2.value.n_rows, t2.value.n_cols);
                return std::make_pair(t1_broadcasted, t2_broadcasted);
            }

            if (t2.is_scalar()) {
                t2_broadcasted = arma::repmat(t2_broadcasted, t1.value.n_rows, t1.value.n_cols);
                return std::make_pair(t1_broadcasted, t2_broadcasted);
            }

            if (t1.value.n_cols == t2.value.n_rows) {
                t2_broadcasted = t2_broadcasted.t();
            }

            // if (t2.is_col_vector()) {
            //     cout << "t2 col vector\n";
            //     t2_broadcasted = arma::repmat(t2_broadcasted, 1, t1.value.n_cols);
                
            // }

            return std::make_pair(t1_broadcasted, t2_broadcasted);
        }

        template<typename T, typename Op>
        Tensor<T> applyTensorElementWiseOperation(const Tensor<T>& l, const Tensor<T>& r, Op operation) {
            auto [broadcasted_l, broadcasted_r] = element_wise_broadcast(l, r);

            return Tensor<T>(operation(broadcasted_l, broadcasted_r));
        }

        

        /// -----------------------
        /// Element-wise Operations
        /// -----------------------

        template<class T>
        struct power {
            T operator()(const T& base, const T& exponent) const { return arma::pow(base, exponent); }
        };

        template<typename T> Tensor<T> operator+(const Tensor<T>& x) { return x; }
        template<typename T> Tensor<T> operator-(const Tensor<T>& x) { return Tensor<T>(-x.value); }

        template<typename T> Tensor<T> operator+(const Tensor<T>& l, const Tensor<T>& r) { return applyTensorElementWiseOperation(l, r, std::plus<mat<T>>()); }
        template<typename T> Tensor<T> operator-(const Tensor<T>& l, const Tensor<T>& r) { return applyTensorElementWiseOperation(l, r, std::minus<mat<T>>()); }
        template<typename T> Tensor<T> operator/(const Tensor<T>& l, const Tensor<T>& r) { return applyTensorElementWiseOperation(l, r, std::divides<mat<T>>()); }
        template<typename T> Tensor<T> operator%(const Tensor<T>& l, const Tensor<T>& r) { return applyTensorElementWiseOperation(l, r, std::modulus<mat<T>>()); }
        template<typename T> Tensor<T> pow(const Tensor<T>& l, const Tensor<T>& r) { return applyTensorElementWiseOperation(l, r, power<mat<T>>()); }


        template<typename T, typename U, Requires<IsArithmetic<U>> = true> Tensor<T> operator+(const Tensor<T>& l, const U& r) { return l + Tensor<T>(r); }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> Tensor<T> operator-(const Tensor<T>& l, const U& r) { return l - Tensor<T>(r); }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> Tensor<T> operator/(const Tensor<T>& l, const U& r) { return l / Tensor<T>(r); }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> Tensor<T> operator%(const Tensor<T>& l, const U& r) { return l % Tensor<T>(r); }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> Tensor<T> pow(const Tensor<T>& l, const U& r) { return pow(l, Tensor<T>(r)); }

        template<typename T, typename U, Requires<IsArithmetic<U>> = true> Tensor<T> operator+(const U& l, const Tensor<T>& r) { return Tensor<T>(l) + r; }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> Tensor<T> operator-(const U& l, const Tensor<T>& r) { return Tensor<T>(l) - r; }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> Tensor<T> operator/(const U& l, const Tensor<T>& r) { return Tensor<T>(l) / r; }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> Tensor<T> operator%(const U& l, const Tensor<T>& r) { return Tensor<T>(l) % r; }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> Tensor<T> pow(const U& l, const Tensor<T>& r) { return pow(Tensor<T>(l), r); }

        /// ---------------------------
        /// Matrix multiplication, dot product, etc.
        /// ---------------------------

        template<typename T> Tensor<T> operator*(const Tensor<T>& l, const Tensor<T>& r) {
            
            // cout << l << endl;
            // cout << r << endl;

            if (l.is_scalar()) {
                return Tensor<T>(l(0,0) * r.value);
            }

            if (r.is_scalar()) {
                return Tensor<T>(l.value * r(0,0));
            }

            if (r.is_col_vector() && l.is_col_vector()) {
                return Tensor<T>(l.value * r.value.t());
            }

            return Tensor<T>(l.value * r.value);

        }

        template<typename T, typename U, Requires<IsArithmetic<U>> = true> Tensor<T> operator*(const Tensor<T>& l, const U& r) { return Tensor<T>(l.value * r); }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> Tensor<T> operator*(const U& l, const Tensor<T>& r) { return Tensor<T>(l * r.value); }

        /// -----------------------
        /// Trigonometric Functions
        /// -----------------------
        template<typename T> Tensor<T> sin(const Tensor<T>& x) { return Tensor<T>(arma::sin(x.value)); }
        template<typename T> Tensor<T> cos(const Tensor<T>& x) { return Tensor<T>(arma::cos(x.value)); }
        template<typename T> Tensor<T> tan(const Tensor<T>& x) { return Tensor<T>(arma::tan(x.value)); }

        /// ---------------
        /// Other functions
        /// ---------------
        template<typename T> Tensor<T> abs(const Tensor<T>& x) { return Tensor<T>(arma::abs(x.value)); }
        template<typename T> Tensor<T> log(const Tensor<T>& x) { return Tensor<T>(arma::log(x.value)); }
        template<typename T> Tensor<T> mean(const Tensor<T>& x) { return Tensor<T>(arma::mean(x.value)); }

        template<typename T> Tensor<T> stddev(const Tensor<T>& x) { return Tensor<T>(arma::stddev(x.value)); }
    }

}
