#include "fileinfo.hpp"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

int
fileinfo_read(std::vector<fileinfo_t> &list,
			  const char *filename)
{
	FILE *fp;
	char line[8192];
	
	list.clear();
		
	fp = fopen(filename, "r");
	if (fp == NULL) {
		fprintf(stderr, "fopen failed: %s\n", filename);
		return -1;
	}
	
	while (fgets(line, sizeof(line) - 1, fp)) {
		const char *sep = strstr(line, " ");
		if (sep) {
			// labeled data
			const char *file = sep + 1;
			int label = atoi(line);
			size_t len = strlen(line);
			fileinfo_t info;
			
			line[len-1] = '\0';
			
			info.file = file;
			info.label = label;
			list.push_back(info);
		} else {
			const char *file = line;
			size_t len = strlen(file);
			fileinfo_t info;
			
			line[len-1] = '\0';
			
			info.file = file;
			info.label = -1;
			list.push_back(info);
		}
	}
	fclose(fp);

	return 0;
}

const char *
fileinfo_label2name(int label)
{
	static const char *s_map[] = {
		"",
		""
	};
	if (label >= 0) {
		return s_map[label];
	} else {
		fprintf(stderr, "unknown label\n");
		return NULL;
	}
}
