#include "nv_core.h"
#include "nv_io.h"
#include "nv_num.h"
#include "nv_ml.h"

int
main(void)
{
	nv_matrix_t *data = nv_load_matrix_bin("train_data.mat");
	nv_matrix_t *labels = nv_load_matrix_bin("train_labels.mat");
	nv_matrix_t *test_data = nv_load_matrix_bin("test_data.mat");
	nv_matrix_t *test_labels = nv_load_matrix_bin("test_labels.mat");
	int i, ok;
	int k = 0;
	nv_lr_t *lr;
	
	printf("train: %d, %ddim\ntest: %d\n",
		   data->m,
		   data->n,
		   test_data->m
		);
	ok = 0;

	for (i = 0; i < labels->m; ++i) {
		k = NV_MAX(k, NV_MAT_VI(labels, i, 0));
	}
	k += 1;
	lr = nv_lr_alloc(data->n, k);
	nv_lr_progress(1);
	nv_lr_init(lr, data);
	nv_lr_train(lr,
				data, labels,
				NV_LR_PARAM(300, 0.0001f, NV_LR_REG_L2, 0.01f, 0));
				//NV_LR_PARAM(100, 0.1e-10f, NV_LR_REG_L2, 0.01, 0));				
	ok = 0;
	for (i = 0; i < test_data->m; ++i) {
		if (nv_lr_predict_label(lr, test_data, i) == NV_MAT_VI(test_labels, i, 0)) {
			++ok;
		}
	}
	printf("Accuracy = %f%% (%d/%d)\n",
		   (float)ok / test_data->m * 100.0f,
		   ok, test_data->m);
	nv_matrix_free(&data);
	nv_matrix_free(&labels);
	nv_matrix_free(&test_data);
	nv_matrix_free(&test_labels);
	nv_lr_free(&lr);
	
	fflush(stdout);

	return 0;
}
