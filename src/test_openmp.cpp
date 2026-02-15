#include <iostream>
#include <omp.h>

int main() {
    std::cout << "OpenMP Test\n";
    std::cout << "Max threads available: " << omp_get_max_threads() << "\n";

    #pragma omp parallel
    {
        #pragma omp critical
        {
            std::cout << "Hello from thread " << omp_get_thread_num() 
                      << " of " << omp_get_num_threads() << "\n";
        }
    }

    // Simple parallel sum test
    const int N = 1000000;
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; ++i) {
        sum += 1.0 / (1.0 + i);
    }

    std::cout << "Parallel reduction sum: " << sum << "\n";
    std::cout << "OpenMP is working!\n";

    return 0;
}
