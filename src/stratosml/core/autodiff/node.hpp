#pragma once

#include <stratosml/core/autodiff/tensor.hpp>

namespace stratos {
    
    namespace autodiff {

        // Abstract Node 
        template<typename T>
        struct Node {

            Tensor<T> val;

            Node(const Tensor<T>& v) : val(v) {}

            virtual void derive(const Tensor<T>&) = 0;
        };

        template<typename T>
        using NodePtr = std::shared_ptr<Node<T>>;

        // Abstract Node with gradient
        template<typename T>
        struct VariableNode : Node<T> {
            Tensor<T> grad;

            // Get tensor value and build from its shape a gradient tensor
            VariableNode(const Tensor<T>& v) : Node<T>(v) {
                grad = Tensor<T>(v, arma::fill::zeros);
            }
        };

        // Node with gradient and without ancestors.
        template<typename T>
        struct IndependentVariableNode : VariableNode<T> {

            using VariableNode<T>::grad;

            IndependentVariableNode(const Tensor<T>& v) : VariableNode<T>(v) {}

            void derive(const Tensor<T>& grad) override {
                this->grad += grad;
            }
        };

        // Node with gradient and with ancestors.
        template<typename T>
        struct DependentVariableNode : VariableNode<T> {

            using VariableNode<T>::grad;

            NodePtr<T> expr;

            DependentVariableNode(const NodePtr<T>& e) : VariableNode<T>(e->val), expr(e)  {}

            void derive(const Tensor<T>& grad) override {
                this->grad += grad;
                this->expr->derive(grad);
            }
        };

        /// Constant node i.e. without gradient
        template<typename T>
        struct ConstantNode : Node<T> {

            using Node<T>::Node;

            void derive(const Tensor<T>& grad) override {}
        };

        template<typename T> class Variable;
        template<typename T> class Constant;

        // Abstract common class for Constant and Variable classes
        template<typename T>
        class ConstantOrVariable {
        
        public:
            NodePtr<T> expr;

            ConstantOrVariable(const NodePtr<T>& expr) : expr(expr) {}

            operator class Variable<T>() {
                return Variable<T>(expr);
            }

            void print(std::string prefix) {
                cout << prefix << expr->val;
            }
        };

        template<typename T>
        class Constant : public ConstantOrVariable<T> {

        public:
            using ConstantOrVariable<T>::ConstantOrVariable;
            using ConstantOrVariable<T>::expr;

            Constant(T v): ConstantOrVariable<T>(std::make_shared<ConstantNode<T>>(Tensor(v))) {}

            Constant(const Tensor<T>& x) : ConstantOrVariable<T>(std::make_shared<ConstantNode<T>>(x)) {}

            Constant(std::initializer_list<T> v) : ConstantOrVariable<T>(std::make_shared<ConstantNode<T>>(Tensor(v))) {}

            Constant(std::initializer_list<std::initializer_list<T>> v) : ConstantOrVariable<T>(std::make_shared<ConstantNode<T>>(Tensor(v))) {}

            // Constant(const NodePtr<T>& e) : expr(std::make_shared<DependentVariableNode<T>>(e)) {}

            const std::shared_ptr<ConstantNode<T>> operator->() const {
                return std::dynamic_pointer_cast<ConstantNode<T>>(expr);
            }

            
        };

        template<typename T>
        class Variable : public ConstantOrVariable<T> {


        public:
            using ConstantOrVariable<T>::ConstantOrVariable;
            using ConstantOrVariable<T>::expr;

            Variable() {}

            Variable(const TensorShape& shape): ConstantOrVariable<T>(std::make_shared<IndependentVariableNode<T>>(Tensor<T>(shape))) {}

            Variable(const T& v): ConstantOrVariable<T>(std::make_shared<IndependentVariableNode<T>>(Tensor(v))) {}

            Variable(const Tensor<T>& x) : ConstantOrVariable<T>(std::make_shared<IndependentVariableNode<T>>(x)) {}

            Variable(const std::initializer_list<T>& v) : ConstantOrVariable<T>(std::make_shared<IndependentVariableNode<T>>(Tensor(v))) {}

            Variable(const std::initializer_list<std::initializer_list<T>>& v) : ConstantOrVariable<T>(std::make_shared<IndependentVariableNode<T>>(Tensor(v))) {}

            Variable(const NodePtr<T>& e) : ConstantOrVariable<T>(std::make_shared<DependentVariableNode<T>>(e)) {}

            const std::shared_ptr<VariableNode<T>> operator->() const {
                return std::dynamic_pointer_cast<VariableNode<T>>(expr);
            }

            /// Variable assignment operators

            Variable& operator-=(const Tensor<T>& x) { *this = Variable(expr->val - x); return *this; }

            Variable& operator-=(const NodePtr<T>& x) { *this = Variable(expr - x); return *this; }
            Variable& operator-=(const ConstantOrVariable<T>& x) {
                *this -= x.expr;
                return *this;
            }

            Variable& operator+=(const Tensor<T>& x) { *this = Variable(expr->val + x); return *this; }

            Variable& operator+=(const NodePtr<T>& x) { *this = Variable(expr + x); return *this; }
            Variable& operator+=(const ConstantOrVariable<T>& x) {
                *this += x.expr;
                // expr->val += x.expr->val;
                return *this;
            }

        };

        using const_or_var = ConstantOrVariable<float>;

        using var = Variable<float>;
        using constant = Constant<float>;

        using var_d = Variable<double>;
        using constant_d = Constant<double>;


    }

}