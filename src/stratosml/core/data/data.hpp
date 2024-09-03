#pragma once
#include <iostream>
#include <armadillo>
#include <vector>
#include <memory>
#include <map>
#include <unordered_map>
#include <ranges>
#include <iomanip>
#include <variant>

#include <stratosml/core/autodiff/tensor.hpp>

// using namespace std;

using namespace stratos::autodiff;

// using namespace arma;

namespace stratos {
    namespace data {

        enum class Scaler {
            MaxAbs,
            MinMax,
            Standart,
            Mean
        };



        class Series {

            friend class DataFrame;

            std::string name;


            // arma::vec data;

        public:
            Tensor<float> data;

            Series() {}

            Series(string name): name(name) {}
            Series(string name, size_t size): name(name) {
                data = Tensor<float>(TensorShape({ size }));
            }

            /* Operators */

            float& operator[](size_t index) {
                return data(index);
            }

            const float& operator[](size_t index) const {
                return data(index);
            }

            Series& operator/(float i) {
                data = data / i;
                return *this;
            }

            std::string GetName() {
                return name;
            }

            size_t GetSize() const {
                return data.shape[0];
            }

            void Scale(Scaler scaler) {
                switch (scaler) {
                    case Scaler::MaxAbs:
                        data = data / abs(data).max();
                        break;
                    case Scaler::MinMax:
                        data = (data - data.min()) / (data.max() - data.min());
                        break;
                    case Scaler::Standart:
                        data = (data - mean(data)) / stddev(data);
                        break;
                    case Scaler::Mean:
                        data = (data - mean(data)) / (data.max() - data.min());
                        break;
                    default:
                        throw std::invalid_argument("Invalid scaler.");
                }
            }

            friend std::ostream& operator<<(ostream& stream, const Series& series) {
                stream  << left << setw(7) << " ";
                stream << left << setw(20) << series.name;
                stream << endl;

                for (size_t row = 0; row < series.data.shape.dims[0]; row++) {
                    stream  << left << setw(7) << row;
                    stream << left << setw(20) << series[row];
                    stream << endl;
                }

                return stream;
            }
        };

        class DataFrame {

            std::unordered_map<std::string, Series> data;
            std::vector<string> columns;
            std::pair<size_t, size_t> shape;

        public:

            float& operator()(size_t row, size_t col) {
                return data.at(columns.at(col))[row];
            }

            const float& operator()(size_t row, size_t col) const {
                return data.at(columns.at(col))[row];
            }


            Series& operator[](const string& col) {
                bool column_exists = data.contains(col);

                if (!column_exists)
                    throw std::invalid_argument("Column does not exist");

                return data[col];
            }

            // vector<Series*> operator[](string cols[]) {
            //     int n_cols = sizeof(cols) / sizeof(cols[0]);
            //     vector<Series*> series(n_cols);

            //     for (int i = 0; i < n_cols; i++) {
            //         series.push_back(&data.at(cols[i]));
            //     }

            //     return series;
            // }

            std::vector<std::string> GetColumns() {
                return columns;
            }

            
            std::pair<size_t, size_t> GetShape() const {
                return std::make_pair<size_t, size_t>(data.at(columns.at(0)).GetSize(), columns.size());
            }


            void AddColumn(const string name) {
                Series *series = new Series(name);
                data[name] = *series;
                columns.push_back(name);
            }

            void AddColumn(const string name, size_t size) {
                Series *series = new Series(name, size);
                data[name] = *series;
                columns.push_back(name);
            }

            void AddColumn(const Series& column) {
                data[column.name] = column;
                columns.push_back(column.name);
            }

            void RemoveColumn(const string name) {
                auto pos = std::find(columns.begin(), columns.end(), name);
                if (pos == columns.end())
                    throw "RemoveColumn: Column not found.";
                
                columns.erase(pos);
            }

            void Scale(Scaler scaler) {
                for (auto [name, series]: data) {
                    series.Scale(scaler);
                }
            }

            friend std::ostream& operator<<(ostream& stream, const DataFrame& df) {
                using namespace std;

                stream << left << setw(7) << " ";

                for (string label : df.columns) {
                    stream << left << setw(20) << label;
                }
                stream << "\n";

                auto [rows, cols] = df.GetShape();

                for (size_t row = 0; row < rows; row++) {
                    stream  << left << setw(7) << row;
                    for (size_t col = 0; col < cols; col++) {
                         stream << left << setw(20) << df(row, col);
                    }
                    stream << "\n";
                }

                return stream;
            }

        };

        

        // class SeriesProxy : public Series {
            
        // };

        // template<typename T>
        // struct is_dataframe_or_series : std::false_type {};

        // template<>
        // struct is_dataframe_or_series<DataFrame> : std::true_type {};

        // template<>
        // struct is_dataframe_or_series<Series> : std::true_type {};


        // template<typename T>
        // concept DataFrameOrSeries = std::is_same_v<T, DataFrame> || std::is_same_v<T, Series>;



        // template<typename... DataFrames, typename... Args>
        // typename std::enable_if<(is_dataframe_or_series<DataFrames>::value && ...), std::tuple<DataFrames...>>::type
        // Split(
        //     const std::tuple<DataFrames...>& dfs,
        //     const float test_ratio = 0.25f,
        //     const size_t random_state,
        //     const bool shuffle = true,)
        // {
        //     cout << sizeof(dfs);
        //     return false;
        // }

        bool LoadCSV(fstream& stream, DataFrame& df);

        bool Load(const string& filename, DataFrame& df) {

            
            const int ext_pos = filename.rfind(".");

            if (ext_pos == string::npos) {
                cerr << "Error loading the dataset: no extension." << endl;
                return false;
            }

            const string ext = filename.substr(ext_pos + 1, filename.length());
            
            if (ext != "csv") {
                cerr << "Error loading the dataset: unknown extension." << endl;
                return false;
            }
            
            fstream stream;

            stream.open(filename, fstream::in);

            if (!stream.is_open()) { 
                cerr << "Error loading the dataset!" << endl; 
                return false; 
            } 

            cout << "Successfully loaded the dataset!" << endl;

            if (ext == "csv") {
                if (!LoadCSV(stream, df)) {
                    cerr << "Error loading the dataset: failed reading csv format." << endl; 
                    return false; 
                }
            }

            stream.close();

            return true;
        };

        vector<string> GetColumnLabels(fstream& f) {

            vector<string> labels;

            string first_line;
            getline(f, first_line);

            stringstream linestream(first_line);

            string curr_label;
            while (getline(linestream, curr_label, ',')) {
                labels.push_back(curr_label);
            }

            return labels;

        }

        pair<size_t, size_t> GetMatrixSize(fstream& f) {

            size_t n_rows = 0;
            size_t n_cols = 0;
            string line;
            string token;

            stringstream linestream;

            f.clear();
            streampos start_pos = f.tellg();

            while (f.good()) {
                
                getline(f, line);

                if (line.size() == 0) break;
                
                linestream = stringstream(line);

                size_t curr_cols = 0;

                while (getline(linestream, token, ',')) {
                    curr_cols++;
                }

                if (curr_cols > n_cols) {
                    n_cols = curr_cols;
                }

                n_rows++;
            }

            f.clear();
            f.seekg(start_pos);

            return make_pair(n_rows, n_cols);
        }

        bool LoadCSV(fstream& f, DataFrame& df) {

            vector<string> columns = GetColumnLabels(f);
            pair<int, int> size = GetMatrixSize(f);

            for (string col : columns) {
                df.AddColumn(col, size.first);
            }

            size_t row = 0;
            string line;

            while (getline(f, line)) {
                istringstream str_stream(line);
                string value;

                size_t col = 0;

                while (getline(str_stream, value, ',')) {
                    df(row, col) = stof(value);
                    col++;
                }

                row++;
            }

            return true;
        }
        

    }

}