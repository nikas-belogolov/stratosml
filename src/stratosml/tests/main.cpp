#include <stratosml/core.hpp>
#include <armadillo>

using namespace stratos;
using namespace std;
using namespace stratos::autodiff;
using namespace stratos::layers;
using namespace stratos::data;

int main() {

    const double learning_rate = 0.001;
    const size_t epochs = 10;

    data::DataFrame df;
    data::Load("datasets/Salary_dataset.csv", df);

    df.RemoveColumn("");


    df["YearsExperience"].Scale(Scaler::MaxAbs);
    df["Salary"].Scale(Scaler::MaxAbs);

    cout << df;


    Model model;

    model.Add(new Dense(1, "First Layer"));

    model.optimizer = new GradientDescent(new StepDecay(0.001, 0.9, 2));

    model.Fit(df["YearsExperience"], df["Salary"], epochs);

    constant x_test = {0.754717};

    auto pred = model.Predict(x_test);

    pred.print("Predictions: ");
}