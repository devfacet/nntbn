#define _POSIX_C_SOURCE 199309L

#include "./argmax_perf.h"
#include "nn_app.h"
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
    nn_init_app(argc, argv);
    // nn_set_debug_level(5); // for debugging

    srand((unsigned int)time(NULL));

    test_nn_argmax();

    return 0;
}
