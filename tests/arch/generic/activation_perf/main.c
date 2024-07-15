#include "./activation_perf.h"
#include "nn_app.h"
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    nn_init_app(argc, argv);
    // nn_set_debug_level(5); // for debugging

    srand((unsigned int)time(NULL));

    test_nn_act_func_init();
    test_nn_act_func();
    test_nn_act_func_identity();
    test_nn_act_func_sigmoid();
    test_nn_act_func_relu();
    test_nn_act_func_softmax();
    test_nn_act_func_scalar_batch();
    test_nn_act_func_tensor_batch();

    return 0;
}
