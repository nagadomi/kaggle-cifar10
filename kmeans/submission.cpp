#include "nv_core.h"
#include "nv_ip.h"
#include "nv_io.h"
#include "nv_num.h"
#include "nv_ml.h"
#include "fileinfo.hpp"
#include "param.h"

#define LOAD_DATA 0
#define LOAD_MLP 0
#define TRAIN_MLP 0

static void
extract_features(int augmentation,
				 nv_matrix_t *data,
				 nv_matrix_t *labels,
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
		if (labels) {
			NV_MAT_V(labels, i, 0) = (float)list[i].label;
		}
		if (augmentation) {
			nv_matrix_t *src2 = nv_matrix_clone(src);
			nv_flip_x(src2, src);
			kmeans_feature(data, list.size() + i, src2, zca_m, zca_u, centroids);
			if (labels) {
				NV_MAT_V(labels, list.size() + i, 0) = (float)list[i].label;
			}
			nv_matrix_free(&src2);
		}
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
	nv_matrix_t *train_data = NULL;
	nv_matrix_t *train_labels = NULL;
	nv_matrix_t *test_data = NULL;
	nv_mlp_t *mlp = NULL;
	float ir, hr;
	int i;
	FILE *fp;
	static const char *label_name[10] = {
		"frog",
		"truck",
		"deer",
		"automobile",
		"bird",
		"horse",
		"ship",
		"cat",
		"dog",
		"airplane"
	};
	
	std::vector<fileinfo_t> train_list, test_list;
#if LOAD_DATA
#if TRAIN_MLP
	train_data = nv_load_matrix_bin("train_data.mat");
	train_labels = nv_load_matrix_bin("train_labels.mat");
#endif
	test_data = nv_load_matrix_bin("test_data.mat");
#else
	fileinfo_read(train_list, TRAIN_FILE);
	fileinfo_read(test_list, TEST_FILE);
	printf("read file list train:%d, test:%d\n",
		   (int)train_list.size(), (int)test_list.size());

	train_data = nv_matrix_alloc(DATA_N, train_list.size() * 2);
	train_labels = nv_matrix_alloc(1, train_list.size() * 2);
	test_data = nv_matrix_alloc(DATA_N, test_list.size());
	
	extract_features(1, train_data, train_labels, train_list,
					 zca_m, zca_u, centroids);
	extract_features(0, test_data, NULL, test_list,
					 zca_m, zca_u, centroids);
	
	printf("end read_data\n");
	nv_standardize_train(sd_m, 0, sd_sd, 0, train_data, 0.01f);
	nv_standardize_all(train_data, sd_m, 0, sd_sd, 0);
	nv_standardize_all(test_data, sd_m, 0, sd_sd, 0);
	
	nv_save_matrix_bin("train_data.mat", train_data);
	nv_save_matrix_bin("train_labels.mat", train_labels);
	nv_save_matrix_bin("test_data.mat", test_data);
	printf("end standardize\n");
#endif
	if (train_data && test_data) {
		printf("train: %d, test: %d, dim:%d\n",
			   train_data->m, test_data->m, train_data->n);
	}
#if LOAD_MLP
	mlp = nv_load_mlp("epoch_1.mlp");
	printf("%p\n", mlp);
	printf("%d %d %d %f\n", mlp->input, mlp->hidden, mlp->output, mlp->dropout);
#if TRAIN_MLP
	nv_mlp_dropout(mlp, 0.5f);
	nv_mlp_noise(mlp, 0.2f);
	nv_mlp_progress(1);
	ir = 0.0002f;
	hr = 0.0002f;
	for (i = TRAIN_MLP; i < 3; ++i) {
		char file[256];
		
		if (i >= 2) {
			ir = 0.00005f;
			hr = 0.00005f;
		}
		nv_mlp_train_ex(mlp, train_data, train_labels, ir, hr,
						i * 50, (1 + i) * 50, 150);
		nv_snprintf(file, sizeof(file), "epoch_%d.mlp", i);
		nv_save_mlp(file, mlp);
	}
#endif
#else
	mlp = nv_mlp_alloc(train_data->n, HIDDEN_UNIT, CLASS);
	nv_mlp_init(mlp, train_data);
	nv_mlp_dropout(mlp, 0.5f);
	nv_mlp_noise(mlp, 0.2f);
	nv_mlp_progress(1);
	ir = 0.0002f;
	hr = 0.0002f;
	for (i = 0; i < 5; ++i) {
		char file[256];
		
		if (i >= 4) {
			ir = 0.00005f;
			hr = 0.00005f;
		}
		nv_mlp_train_ex(mlp, train_data, train_labels, ir, hr,
						i * 100, (1 + i) * 100, 500);
		nv_snprintf(file, sizeof(file), "epoch_%d.mlp", i);
		nv_save_mlp(file, mlp);
	}
#endif
	fp = fopen("submission.txt", "w");
	fprintf(fp, "id,label\n");
	for (i = 0; i < test_data->m; ++i) {
		fprintf(fp, "%d,%s\n",
				i + 1, label_name[nv_mlp_predict_label(mlp, test_data, i)]);
	}
	fclose(fp);
	
	nv_matrix_free(&zca_u);
	nv_matrix_free(&zca_m);
	nv_matrix_free(&sd_m);
	nv_matrix_free(&sd_sd);
	nv_matrix_free(&centroids);
	nv_matrix_free(&train_data);
	nv_matrix_free(&train_labels);
	nv_matrix_free(&test_data);
	nv_mlp_free(&mlp);
	
	return 0;
}
