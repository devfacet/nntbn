#include "./accuracy.h"
#include "nn_app.h"

int main(int argc, char *argv[]) {
    nn_init_app(argc, argv);
    // nn_set_debug_level(5); // for debugging

    test_nn_accuracy();

    return 0;
}
