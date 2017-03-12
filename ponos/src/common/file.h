#ifndef PONOS_COMMON_FILE_H
#define PONOS_COMMON_FILE_H

namespace ponos {

/** \brief loads contents from file
 *
 * @param filename **[in]** path/to/file.
 * @param text     **[out]** receives file content.
 *
 * \return number of bytes successfuly read.
 */
int readFile(const char *filename, char **text);

} // ponos namespace

#endif // PONOS_COMMON_FILE_H
