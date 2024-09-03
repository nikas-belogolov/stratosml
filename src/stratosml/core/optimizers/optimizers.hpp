
#pragma once
#include <armadillo>
#include <cmath>
#include <stratosml/core/autodiff/autodiff.hpp>
#include <stratosml/core/optimizers/schedules.hpp>

// using namespace arma;
using namespace stratos::autodiff;
using namespace stratos::optimizers::schedules;

namespace stratos {

    

    namespace optimizers {

        class Optimizer {

        public:

            LearningRateScheduler* lr_scheduler;

            const double& lr = lr_scheduler->lr;

            Optimizer(float lr) : lr_scheduler(new LearningRateScheduler(lr)) {}

            Optimizer(LearningRateScheduler* lr) : lr_scheduler(lr) {}

            ~Optimizer() {
                delete lr_scheduler;
            }

            virtual void build(const std::vector<std::shared_ptr<var>>& params) {}

            virtual void step(const std::vector<std::shared_ptr<var>>& params) = 0;
        };

        class GradientDescent : public Optimizer {

        public:
            using Optimizer::Optimizer;

            void step(const std::vector<std::shared_ptr<var>>& params) {
                for (int i = 0; i < params.size(); ++i) {
                    auto param = params[i];

                    *param -= lr * (*param)->grad;
                }
            }
        };

        class Momentum : public Optimizer {

            float momentum;
            
            std::vector<var> v;

        public:
            Momentum(float lr, float momentum) : Optimizer(lr), momentum(momentum) {}

            void build(const std::vector<std::shared_ptr<var>>& params) {
                for (const auto& param : params) {
                    v.emplace_back((*param)->val.shape);
                }
            }

            void step(const std::vector<std::shared_ptr<var>>& params) {
                for (int i = 0; i < params.size(); ++i) {
                    auto param = params[i];

                    v[i] = this->momentum * v[i] - lr * (*param)->grad;
                    *param += v[i];
                }
            }
        };

        class Adam : public Optimizer {

            double beta1 = 0.9;
            double beta2 = 0.999;
            double epsilon = std::pow(10, -8);
            size_t t = 0;

            std::vector<var> v;
            std::vector<var> m;

        public:

            Adam(float learning_rate) : Optimizer(learning_rate) {}

            void build(const std::vector<std::shared_ptr<var>>& parameters) {
                for (const auto& param : parameters) {
                    v.emplace_back((*param)->val.shape);
                    m.emplace_back((*param)->val.shape);
                }
            }

            void step(const std::vector<std::shared_ptr<var>>& parameters) {
                t += 1;

                for (int i = 0; i < parameters.size(); ++i) {
                    m[i] = beta1 * m[i] + (1 - beta1) * (*parameters[i])->grad;

                    v[i] = beta2 * v[i] + (1 - beta2) * pow((*parameters[i])->grad, 2);
                }

            }

        };

        // class Adam : public Optimizer {
        //     double beta1 = 0.9;
        //     double beta2 = 0.999;
        //     double epsilon = std::pow(10, -8);
        //     int t = 0;

        //     mat m_w;
        //     mat v_w;
        //     vec m_b;
        //     vec v_b;

        // public:
            
        //     Adam(double learning_rate): Optimizer(learning_rate) {
        //         // this->momentum = momentum;
        //     }

        //     Adam(const Adam& other)
        //         : Optimizer(other) {}

        //     Adam* clone() const override {
        //         return new Adam(*this); // Use the copy constructor
        //     }

        //     void init(uint input_shape, uint units) {
        //         m_w = mat(input_shape, units, fill::zeros);
        //         v_w = mat(input_shape, units, fill::zeros);
        //         m_b = vec(units, fill::zeros);
        //         v_b = vec(units, fill::zeros);
        //     }

        //     void step(const mat& dw, const vec& db, mat& w, vec& b) {
        //         t += 1;

        //         m_w = beta1 * m_w + (1 - beta1) * dw;
        //         m_b = beta1 * m_b + (1 - beta1) * db;

        //         v_w = beta2 * v_w + (1 - beta2) * arma::square(dw);
        //         v_b = beta2 * v_b + (1 - beta2) * arma::square(db);

        //         mat m_w_hat = m_w / (1 - std::pow(beta1, t));
        //         vec m_b_hat = m_b / (1 - std::pow(beta1, t));

        //         mat v_w_hat = v_w / (1 - std::pow(beta2, t));
        //         vec v_b_hat = v_b / (1 - std::pow(beta2, t));
                
        //         w -= learning_rate * m_w_hat / (arma::sqrt(v_w_hat) + epsilon);
        //         b -= learning_rate * m_b_hat / (arma::sqrt(v_b_hat) + epsilon);
        //     }
        // };


    }
}