#pragma once
#include <armadillo>
#include <cmath>

// using namespace arma;

namespace stratos {
    
    namespace optimizers {

        namespace schedules {

            class LearningRateScheduler {

            
            protected:
            
            public:
                double lr;

                LearningRateScheduler(double initial_lr) : lr(initial_lr) {
                    if (initial_lr <= 0) throw std::invalid_argument("Initial learning rate must be above zero.");
                }

                virtual void step(size_t epoch) {

                }

            };

            class StepDecay : public LearningRateScheduler {

                double decay_rate;
                size_t step_size;
                double initial_lr;

            public:
                
                StepDecay(double lr, double decay_rate, size_t step_size) : LearningRateScheduler(lr), initial_lr(lr), decay_rate(decay_rate), step_size(step_size) {
                    if (decay_rate <= 0 || decay_rate >= 1) throw std::invalid_argument("Decay rate must be above zero and below one.");
                    if (step_size <= 0) throw std::invalid_argument("Step size must be above zero.");
                }

                void step(size_t epoch) override {
                    this->lr = initial_lr * std::pow(decay_rate, (epoch / step_size)); 
                }

            };

            class ExponentialDecay : public LearningRateScheduler {

                double decay_rate;
                double initial_lr;

            public:

                ExponentialDecay(double lr, double decay_rate) : LearningRateScheduler(lr), decay_rate(decay_rate), initial_lr(lr) {
                    if (decay_rate <= 0 || decay_rate >= 1) throw std::invalid_argument("Decay rate must be above zero and below one.");
                }

                void step(size_t epoch) override {
                    this->lr = initial_lr * std::exp(-decay_rate * epoch);
                }

            };

        }
    }
}