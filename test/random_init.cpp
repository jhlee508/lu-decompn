#include <iostream>
#include <vector>
#include <cstdlib>

using namespace std;

int main() {
    srand(42);
    int matSize = 3; // Change the matrix size as needed

    // const std::vector<double> A = 
    //     {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0};
    vector<double> A(matSize * matSize);
    for (int i = 0; i < matSize; ++i) {
        for (int j = 0; j < matSize; ++j) {
            double element = (double)1.0 + ((double)rand() / (double)RAND_MAX);
            A[j * matSize + i] = element;
        }
    }

    cout << "Randomly initialized matrix-like vector A:" << endl;
    for (int i = 0; i < matSize; ++i) {
        for (int j = 0; j < matSize; ++j) {
            cout << A[j * matSize + i] << " ";
        }
        cout << endl;
    }

    return 0;
}