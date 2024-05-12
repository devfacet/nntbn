#ifndef NN_APP_H
#define NN_APP_H

#include <stdbool.h>

/**
 * @brief Initializes the running app with the given command line arguments.
 *
 * @param argc The number of command line arguments.
 * @param argv The command line arguments.
 *
 * @example
 *  int main(int argc, char *argv[]) {
 *      nn_init_app(argc, argv);
 *      return 0;
 *  }
 */
void nn_init_app(int argc, char *argv[]);

/**
 * @brief Parses the given command line arguments.
 *
 * @param argc The number of command line arguments.
 * @param argv The command line arguments.
 */
void nn_parse_args(int argc, char *argv[]);

/**
 * @brief Looks up a specific flag.
 *
 * @param flag The flag to look up.
 */
bool nn_lookup_flag(const char *flag);

/**
 * @brief Returns the value of a specific flag.
 *
 * @param flag The flag to get.
 */
const char *nn_get_flag(const char *flag);

#endif // NN_APP_H
