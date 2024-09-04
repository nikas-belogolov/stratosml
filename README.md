
# STRATOSML - C++ Machine Learning Library


## Example

```c++
#include <stratosml/core.hpp>

using namespace stratos;
using namespace stratos::optimizers;
using namespace stratos::optimizers::schedules;

int main() {

    data::DataFrame df;
    data::Load("datasets/Salary_dataset.csv", df);

    df.RemoveColumn("");

    df["YearsExperience"].Scale(Scaler::MaxAbs);
    df["Salary"].Scale(Scaler::MaxAbs);

    cout << df;

    const size_t epochs = 10;

    Model model;

    model.Add(new layers::Dense(1, "First Layer"));

    model.optimizer = new GradientDescent(new StepDecay(0.001, 0.9, 2));

    model.Fit(df["YearsExperience"], df["Salary"], epochs);
}
```


## How to compile
```bash
make
g++ -std=c++20 -I./include -L./lib -fcompare-debug-second <path to file>.cpp -o <output file> -larmadillo -lblas -llapack -lstratosml
```

## TODO:
- Categorical data loading
- DataFrame splitting
- Implement batch training
- Implement more layer types, optimizers, schedulers, activation functions, losses.