#include "nv_core.h"
#include "nv_ip.h"
#include "nv_io.h"
#include "nv_num.h"
#include "nv_ml.h"
#include "fileinfo.hpp"
#include "param.h"

void
patch_sampling(nv_matrix_t *samples, std::vector<fileinfo_t> &list)
{
	nv_matrix_t *data = nv_matrix_alloc(PATCH_SIZE * PATCH_SIZE * 3,
										(int)((IMG_SIZE-PATCH_SIZE) * (IMG_SIZE-PATCH_SIZE) * list.size()));
	int data_index = 0;
	int i;
	
	nv_matrix_zero(data);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
	for (i = 0; i < (int)list.size(); ++i) {
		nv_matrix_t *src;
		nv_matrix_t *patches;
		src = nv_load_image(list[i].file.c_str());
		if (!src) {
			fprintf(stderr, "open filed: %s\n", list[i].file.c_str());
			exit(-1);
		}
		
		patches = nv_patch_matrix_alloc(src, PATCH_SIZE);
		nv_patch_extract(patches, src, PATCH_SIZE);
		
#ifdef _OPENMP
#pragma omp critical (patch_sampling)
#endif
		{
			int j;
			for (j = 0; j < patches->m; ++j) {
				nv_vector_copy(data, data_index, patches, j);
				data_index += 1;
			}
		}
		
		nv_matrix_free(&src);
		nv_matrix_free(&patches);
	}
	nv_vector_shuffle(data);
	nv_matrix_m(data, NV_MIN(samples->m, data_index));
	nv_matrix_copy_all(samples, data);
	nv_matrix_free(&data);
}

void
kmeans(nv_matrix_t *centroids,
	   const nv_matrix_t *data)
{
	nv_matrix_t *cluster_labels = nv_matrix_alloc(1, data->m);
	nv_matrix_t *count = nv_matrix_alloc(1, CENTROIDS);

	nv_matrix_zero(count);
	nv_matrix_zero(centroids);
	nv_matrix_zero(cluster_labels);
	
	nv_kmeans_progress(1);
	nv_kmeans(centroids, count, cluster_labels, data, CENTROIDS, 50);

	nv_matrix_free(&cluster_labels);
	nv_matrix_free(&count);
}

int
main(void)
{
	nv_matrix_t *data = nv_matrix_alloc(PATCH_SIZE * PATCH_SIZE * 3, SAMPLES);
	nv_matrix_t *centroids = nv_matrix_alloc(data->n, CENTROIDS);
	nv_matrix_t *zca_u = nv_matrix_alloc(data->n, data->n);
	nv_matrix_t *zca_m = nv_matrix_alloc(data->n, 1);
	std::vector<fileinfo_t> list;
	
	fileinfo_read(list, TRAIN_FILE);
	printf("read file list %d\n", (int)list.size());
	patch_sampling(data, list);
	printf("end patch sampling\n");
	
	nv_standardize_local_all(data, 10.0f);
	nv_zca_train(zca_m, 0, zca_u, data, 0.1f);
	nv_zca_whitening_all(data, zca_m, 0, zca_u);
	printf("end whitening\n");
	kmeans(centroids, data);
	
	nv_save_matrix_bin("zca_u.mat", zca_u);
	nv_save_matrix_bin("zca_m.mat", zca_m);
	nv_save_matrix_bin("centroids.mat", centroids);
	
	nv_matrix_free(&data);
	nv_matrix_free(&zca_u);
	nv_matrix_free(&zca_m);
	nv_matrix_free(&centroids);
	
	return 0;
}
