#pragma once

#include <stratosml/core/autodiff/tensor.hpp>
#include <stratosml/core/autodiff/node.hpp>
#include <armadillo>
#include <initializer_list>

namespace stratos {

    namespace autodiff {

        template<typename T>
        struct UnaryExprNode : Node<T> {
            NodePtr<T> x;

            UnaryExprNode(const Tensor<T>& v, const NodePtr<T>& x) : Node<T>(v), x(x) {}
        };

        template<typename T>
        struct BinaryExprNode : Node<T> {
            NodePtr<T> l, r;

            BinaryExprNode(const Tensor<T>& v, const NodePtr<T>& l, const NodePtr<T>& r) : Node<T>(v), l(l), r(r) {}
        };
        
        template<typename T>
        struct AddExprNode : BinaryExprNode<T> {

            using BinaryExprNode<T>::l;
            using BinaryExprNode<T>::r;
            using BinaryExprNode<T>::BinaryExprNode;

            void derive(const Tensor<T>& grad) override {
                l->derive(grad);
                r->derive(grad);
            }
        };

        template<typename T>
        struct SubExprNode : BinaryExprNode<T> {

            using BinaryExprNode<T>::l;
            using BinaryExprNode<T>::r;
            using BinaryExprNode<T>::BinaryExprNode;

            void derive(const Tensor<T>& grad) override {
                l->derive(grad);
                r->derive(-grad);
            }
        };

        template<typename T>
        struct MulExprNode : BinaryExprNode<T> {

            using BinaryExprNode<T>::l;
            using BinaryExprNode<T>::r;
            using BinaryExprNode<T>::BinaryExprNode;

            void derive(const Tensor<T>& grad) override {
                l->derive(grad * r->val);
                r->derive(grad * l->val);
            }
        };

        template<typename T>
        struct DivExprNode : BinaryExprNode<T> {

            using BinaryExprNode<T>::l;
            using BinaryExprNode<T>::r;
            using BinaryExprNode<T>::BinaryExprNode;

            void derive(const Tensor<T>& grad) override {
                const auto aux1 = 1.0 / r->val;
                const auto aux2 = -l->val * aux1 * aux1;
                l->derive(aux1);
                r->derive(aux2);
            }
        };

        template<typename T>
        struct AbsExprNode : UnaryExprNode<T> {

            using UnaryExprNode<T>::x;
            using UnaryExprNode<T>::UnaryExprNode;

            void derive(const Tensor<T>& grad) override {
                if (x->val < 0) {
                    x->derive(-grad);
                } else if (x->val > 0) {
                    x->derive(grad);
                } else {
                    x->derive(Tensor<T>(0));
                }
            }
        };

        template<typename T>
        struct PowExprNode : BinaryExprNode<T> {

            using BinaryExprNode<T>::l;
            using BinaryExprNode<T>::r;
            using BinaryExprNode<T>::BinaryExprNode;

            void derive(const Tensor<T>& grad) override {
                const auto aux = grad * pow(l->val, r->val - 1); // grad * l^(r-1)
                l->derive(aux * r->val);
                // cout << "l->val" << l->val << endl;
                // cout << "log(l->val)" << log(l->val) << endl;
                const auto auxr = l->val % log(l->val); // l*log(l)
                // cout << auxr << endl;
                r->derive(aux * auxr); // grad * l^(r)*log(l)
            }

        };

        template<typename T>
        struct MeanExprNode : UnaryExprNode<T> {

            using UnaryExprNode<T>::x;
            using UnaryExprNode<T>::UnaryExprNode;

            void derive(const Tensor<T>& grad) override {
                x->derive(grad);
            }
        };  

        template<typename T>
        struct SinExprNode : UnaryExprNode<T> {

            using UnaryExprNode<T>::x;
            using UnaryExprNode<T>::UnaryExprNode;

            void derive(const Tensor<T>& grad) override {
                x->derive(grad * cos(x->val));
            }
        };

        template<typename T>
        struct CosExprNode : UnaryExprNode<T> {

            using UnaryExprNode<T>::x;
            using UnaryExprNode<T>::UnaryExprNode;

            void derive(const Tensor<T>& grad) override {
                x->derive(-grad * sin(x->val));
            }
        };

        template<typename T>
        struct TanExprNode : UnaryExprNode<T> {

            using UnaryExprNode<T>::x;
            using UnaryExprNode<T>::UnaryExprNode;

            void derive(const Tensor<T>& grad) override {
                const auto aux = 1.0 / cos(x->val);
                x->derive(grad * aux * aux);
            }
        };

        // template<typename T>
        // struct DotExprNode : BinaryExprNode<T> {

        //     using BinaryExprNode<T>::l;
        //     using BinaryExprNode<T>::r;
        //     using BinaryExprNode<T>::BinaryExprNode;

            

        // };

        /// -----------------------
        /// Element-wise Operations (+, -, /, %)
        /// -----------------------

        template<typename T> NodePtr<T> operator+(const NodePtr<T>& x) { return x; }
        // template<typename T> NodePtr<T> operator-(const NodePtr<T>& x) { return Tensor<T>(-x.value); }

        template<typename T> NodePtr<T> operator+(const NodePtr<T>& l, const NodePtr<T>& r) { return std::make_shared<AddExprNode<T>>(l->val + r->val, l, r); }
        template<typename T> NodePtr<T> operator-(const NodePtr<T>& l, const NodePtr<T>& r) { return std::make_shared<SubExprNode<T>>(l->val - r->val, l, r); }
        template<typename T> NodePtr<T> operator/(const NodePtr<T>& l, const NodePtr<T>& r) { return std::make_shared<DivExprNode<T>>(l->val / r->val, l, r); }
        template<typename T> NodePtr<T> operator%(const NodePtr<T>& l, const NodePtr<T>& r) { return std::make_shared<MulExprNode<T>>(l->val % r->val, l, r); }
        template<typename T> NodePtr<T> pow(const NodePtr<T>& l, const NodePtr<T>& r) { return std::make_shared<PowExprNode<T>>(pow(l->val, r->val), l, r); }


        template<typename T> NodePtr<T> operator+(const ConstantOrVariable<T>& l, const ConstantOrVariable<T>& r) { return l.expr + r.expr; }
        template<typename T> NodePtr<T> operator-(const ConstantOrVariable<T>& l, const ConstantOrVariable<T>& r) { return l.expr - r.expr; }
        template<typename T> NodePtr<T> operator/(const ConstantOrVariable<T>& l, const ConstantOrVariable<T>& r) { return l.expr / r.expr; }
        template<typename T> NodePtr<T> operator%(const ConstantOrVariable<T>& l, const ConstantOrVariable<T>& r) { return l.expr % r.expr; }
        template<typename T> NodePtr<T> pow(const ConstantOrVariable<T>& l, const ConstantOrVariable<T>& r) { return pow(l.expr, r.expr); }

        template<typename T> NodePtr<T> operator+(const NodePtr<T>& l, const ConstantOrVariable<T>& r) { return l + r.expr; }
        template<typename T> NodePtr<T> operator-(const NodePtr<T>& l, const ConstantOrVariable<T>& r) { return l - r.expr; }
        template<typename T> NodePtr<T> operator/(const NodePtr<T>& l, const ConstantOrVariable<T>& r) { return l / r.expr; }
        template<typename T> NodePtr<T> operator%(const NodePtr<T>& l, const ConstantOrVariable<T>& r) { return l % r.expr; }
        template<typename T> NodePtr<T> pow(const NodePtr<T>& l, const ConstantOrVariable<T>& r) { return pow(l, r.expr); }


        template<typename T> NodePtr<T> operator+(const ConstantOrVariable<T>& l, const NodePtr<T>& r) { return l.expr + r; }
        template<typename T> NodePtr<T> operator-(const ConstantOrVariable<T>& l, const NodePtr<T>& r) { return l.expr - r; }
        template<typename T> NodePtr<T> operator/(const ConstantOrVariable<T>& l, const NodePtr<T>& r) { return l.expr / r; }
        template<typename T> NodePtr<T> operator%(const ConstantOrVariable<T>& l, const NodePtr<T>& r) { return l.expr % r; }
        template<typename T> NodePtr<T> pow(const ConstantOrVariable<T>& l, const NodePtr<T>& r) { return pow(l.expr, r); }


        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> operator+(const ConstantOrVariable<T>& l, const U& r) { return l + Constant<T>(r); }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> operator-(const ConstantOrVariable<T>& l, const U& r) { return l - Constant<T>(r); }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> operator/(const ConstantOrVariable<T>& l, const U& r) { return l / Constant<T>(r); }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> operator%(const ConstantOrVariable<T>& l, const U& r) { return l % Constant<T>(r); }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> pow(const ConstantOrVariable<T>& l, const U& r) { return pow(l, Constant<T>(r)); }

        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> operator+(const U& l, const ConstantOrVariable<T>& r) { return Constant<T>(l) + r; }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> operator-(const U& l, const ConstantOrVariable<T>& r) { return Constant<T>(l) - r; }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> operator/(const U& l, const ConstantOrVariable<T>& r) { return Constant<T>(l) / r; }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> operator%(const U& l, const ConstantOrVariable<T>& r) { return Constant<T>(l) % r; }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> pow(const U& l, const ConstantOrVariable<T>& r) { return pow(Constant<T>(l), r); }


        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> operator+(const NodePtr<T>& l, const U& r) { return l + Constant<T>(r); }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> operator-(const NodePtr<T>& l, const U& r) { return l - Constant<T>(r); }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> operator/(const NodePtr<T>& l, const U& r) { return l / Constant<T>(r); }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> operator%(const NodePtr<T>& l, const U& r) { return l % Constant<T>(r); }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> pow(const NodePtr<T>& l, const U& r) { return pow(l, Constant<T>(r)); }


        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> operator+(const U& l, const NodePtr<T>& r) { return Constant<T>(l) + r; }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> operator-(const U& l, const NodePtr<T>& r) { return Constant<T>(l) - r; }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> operator/(const U& l, const NodePtr<T>& r) { return Constant<T>(l) / r; }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> operator%(const U& l, const NodePtr<T>& r) { return Constant<T>(l) % r; }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> pow(const U& l, const NodePtr<T>& r) { return pow(Constant<T>(l), r); }


        template<typename T> NodePtr<T> operator+(const NodePtr<T>& l, const Tensor<T>& r) { return l + constant(r); }
        template<typename T> NodePtr<T> operator-(const NodePtr<T>& l, const Tensor<T>& r) { return l - constant(r); }


        /// ---------------------------
        /// Non Element-wise Operations
        /// ---------------------------

        template<typename T> NodePtr<T> operator*(const NodePtr<T>& l, const NodePtr<T>& r) { return std::make_shared<MulExprNode<T>>(l->val * r->val, l, r); }

        template<typename T> NodePtr<T> operator*(const ConstantOrVariable<T>& l, const ConstantOrVariable<T>& r) { return l.expr * r.expr; }
        template<typename T> NodePtr<T> operator*(const NodePtr<T>& l, const ConstantOrVariable<T>& r) { return l * r.expr; }
        template<typename T> NodePtr<T> operator*(const ConstantOrVariable<T>& l, const NodePtr<T>& r) { return l.expr * r; }

        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> operator*(const ConstantOrVariable<T>& l, const U& r) { return l * Constant<T>(r); }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> operator*(const U& l, const ConstantOrVariable<T>& r) { return Constant<T>(l) * r; }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> operator*(const NodePtr<T>& l, const U& r) { return l * Constant<T>(r); }
        template<typename T, typename U, Requires<IsArithmetic<U>> = true> NodePtr<T> operator*(const U& l, const NodePtr<T>& r) { return Constant<T>(l) * r; }

        /// -----------------------
        /// Trigonometric Functions
        /// -----------------------
        template<typename T> NodePtr<T> sin(const NodePtr<T>& x) { return std::make_shared<SinExprNode<T>>(sin(x->val), x); }
        template<typename T> NodePtr<T> cos(const NodePtr<T>& x) { return std::make_shared<CosExprNode<T>>(cos(x->val), x); }
        template<typename T> NodePtr<T> tan(const NodePtr<T>& x) { return std::make_shared<TanExprNode<T>>(tan(x->val), x); }

        template<typename T> NodePtr<T> sin(const ConstantOrVariable<T>& x) { return sin(x.expr); }
        template<typename T> NodePtr<T> cos(const ConstantOrVariable<T>& x) { return cos(x.expr); }
        template<typename T> NodePtr<T> tan(const ConstantOrVariable<T>& x) { return tan(x.expr); }

        /// ---------------
        /// Other functions
        /// ---------------
        template<typename T> NodePtr<T> abs(const NodePtr<T>& x) { return std::make_shared<AbsExprNode<T>>(abs(x->val), x); }
        template<typename T> NodePtr<T> abs(const ConstantOrVariable<T>& x) { return abs(x.expr); }

        template<typename T> NodePtr<T> mean(const NodePtr<T>& x) { return std::make_shared<MeanExprNode<T>>(mean(x->val), x); }
        template<typename T> NodePtr<T> mean(const ConstantOrVariable<T>& x) { return mean(x.expr); }
    }

}
