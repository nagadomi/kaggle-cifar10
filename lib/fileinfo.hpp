#ifndef FILEINFO_HPP
#define FILEINFO_HPP

#include <vector>
#include <string>

typedef struct {
	std::string file;
	int label;
} fileinfo_t;

int
fileinfo_read(std::vector<fileinfo_t> &list,
			  const char *filename);

const char *
fileinfo_label2name(int label);

#endif
