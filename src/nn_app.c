#include "nn_app.h"
#include "nn_config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// TODO: Add tests

// Init vars
static int arg_count = 0;
static char **args = NULL;

void nn_init_app(int argc, char *argv[]) {
    nn_parse_args(argc, argv);
    if (nn_get_flag("--debug-level")) {
        nn_set_debug_level(atoi(nn_get_flag("--debug-level")));
    }
}

void nn_parse_args(int argc, char *argv[]) {
    arg_count = argc;
    args = argv;
}

bool nn_lookup_flag(const char *flag) {
    for (int i = 1; i < arg_count; i++) {
        // If the argument starts with the flag and is followed by either '\0' or '=' then
        if (strncmp(args[i], flag, strlen(flag)) == 0 && (args[i][strlen(flag)] == '\0' || args[i][strlen(flag)] == '=')) {
            return true;
        }
    }
    return false;
}

const char *nn_get_flag(const char *flag) {
    size_t flag_len = strlen(flag);
    for (int i = 1; i < arg_count; i++) {
        // If the argument starts with the flag and is followed by '=' then
        if (strncmp(args[i], flag, flag_len) == 0 && args[i][flag_len] == '=') {
            return args[i] + flag_len + 1; // +1 to skip the '='
        }
    }
    return NULL;
}
