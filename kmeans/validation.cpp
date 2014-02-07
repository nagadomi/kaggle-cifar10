#include "nv_core.h"
#include "nv_ip.h"
#include "nv_io.h"
#include "nv_num.h"
#include "nv_ml.h"
#include "fileinfo.hpp"
#include "param.h"

#define TRAIN_M(data) (data->m / 5 * 4)

#define LOAD_DATA 1

static void
validation(const nv_mlp_t *mlp,
		   const nv_matrix_t *test_data,
		   const nv_matrix_t *test_labels)
{
	int i, corret = 0;
	for (i = 0; i < test_data->m; ++i) {
		if (nv_mlp_predict_label(mlp, test_data, i) == NV_MAT_VI(test_labels, i, 0)) {
			++corret;
		}
	}
	printf("Accuracy = %f%% (%d/%d)\n",
		   (float)corret / test_data->m * 100.0f,
		   corret, test_data->m);
}

static void
extract_features(nv_matrix_t *data, nv_matrix_t *labels,
				 const std::vector<fileinfo_t> &list,
				 const nv_matrix_t *zca_m,
				 const nv_matrix_t *zca_u,
				 const nv_matrix_t *centroids)
{
	int i;	
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
	for (i = 0; i < (int)list.size(); ++i) {
		nv_matrix_t *src;
		src = nv_load_image(list[i].file.c_str());
		if (!src) {
			fprintf(stderr, "open filed: %s\n", list[i].file.c_str());
			exit(-1);
		}
		kmeans_feature(data, i, src, zca_m, zca_u, centroids);
		NV_MAT_V(labels, i, 0) = (float)list[i].label;
		nv_matrix_free(&src);
	}
}

int
main(void)
{
	nv_matrix_t *centroids = nv_load_matrix_bin("centroids.mat");
	nv_matrix_t *zca_u = nv_load_matrix_bin("zca_u.mat");
	nv_matrix_t *zca_m = nv_load_matrix_bin("zca_m.mat");
	nv_matrix_t *sd_m = nv_matrix_alloc(DATA_N, 1);
	nv_matrix_t *sd_sd = nv_matrix_alloc(DATA_N, 1);
	nv_matrix_t *data, *labels;
	nv_matrix_t *train_data;
	nv_matrix_t *train_labels;
	nv_matrix_t *test_data;
	nv_matrix_t *test_labels;
	nv_mlp_t *mlp;
	float ir, hr;
	int i;
	
	std::vector<fileinfo_t> list;
#if LOAD_DATA 	
	train_data = nv_load_matrix_bin("train_data.mat");
	train_labels = nv_load_matrix_bin("train_labels.mat");
	test_data = nv_load_matrix_bin("test_data.mat");
	test_labels = nv_load_matrix_bin("test_labels.mat");
#else
	fileinfo_read(list, TRAIN_FILE);
	printf("read file list %d\n", (int)list.size());

	data = nv_matrix_alloc(DATA_N, list.size());
	labels = nv_matrix_alloc(1, list.size());
	train_data = nv_matrix_alloc(data->n, TRAIN_M(data));
	train_labels = nv_matrix_alloc(1, TRAIN_M(data));
	test_data = nv_matrix_alloc(data->n, data->m - train_data->m);
	test_labels = nv_matrix_alloc(1, labels->m - train_labels->m);
	
	extract_features(data, labels, list, zca_m, zca_u, centroids);
	printf("end read_train_data\n");
	nv_dataset(data, labels,
			   train_data, train_labels,
			   test_data, test_labels);
	nv_matrix_free(&data);
	nv_matrix_free(&labels);
	
	nv_standardize_train(sd_m, 0, sd_sd, 0, train_data, 0.01f);
	nv_standardize_all(train_data, sd_m, 0, sd_sd, 0);
	nv_standardize_all(test_data, sd_m, 0, sd_sd, 0);

	nv_save_matrix_bin("train_data.mat", train_data);
	nv_save_matrix_bin("train_labels.mat", train_labels);	
	nv_save_matrix_bin("test_data.mat", test_data);
	nv_save_matrix_bin("test_labels.mat", test_labels);
#endif
	printf("end standardize\n");

	printf("train: %d, test: %d, dim:%d\n", train_data->m, test_data->m, train_data->n);

	mlp = nv_mlp_alloc(train_data->n, HIDDEN_UNIT, CLASS);
	nv_mlp_init(mlp, train_data);
	nv_mlp_dropout(mlp, 0.5f);
	nv_mlp_noise(mlp, 0.2f);
	nv_mlp_progress(1);
	ir = 0.0003f;
	hr = 0.0003f;
	for (i = 0; i < 12; ++i) {
		nv_mlp_train_ex(mlp, train_data, train_labels, ir, hr,
						i * 100, (1 + i) * 100, 1200);
		validation(mlp, test_data, test_labels);
	}
	
	nv_matrix_free(&zca_u);
	nv_matrix_free(&zca_m);
	nv_matrix_free(&sd_m);
	nv_matrix_free(&sd_sd);
	nv_matrix_free(&centroids);
	nv_matrix_free(&train_data);
	nv_matrix_free(&train_labels);
	nv_matrix_free(&test_data);
	nv_matrix_free(&test_labels);
	
	return 0;
}
