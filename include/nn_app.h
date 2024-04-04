#ifndef NN_APP_H
#define NN_APP_H

// nn_init_app initializes an application with the given command line arguments.
void nn_init_app(int argc, char *argv[]);

// nn_parse_args parses the given command line arguments.
void nn_parse_args(int argc, char *argv[]);

// nn_lookup_flag checks if a specific flag exists.
int nn_lookup_flag(const char *flag);

// nn_get_flag returns the value of a specific flag.
const char *nn_get_flag(const char *flag);

#endif // NN_APP_H
