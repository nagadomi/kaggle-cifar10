#include "nv_core.h"
#include "nv_ip.h"
#include "nv_num.h"
#include "param.h"

void
kmeans_feature(nv_matrix_t *fv, int fv_j,
			   const nv_matrix_t *src,
			   const nv_matrix_t *zca_m,
			   const nv_matrix_t *zca_u,
			   const nv_matrix_t *centroids)
{
	nv_matrix_t *patches;
	nv_matrix_t *conv;
	int y, i;	
	
	NV_ASSERT(fv->n == DATA_N);
	patches = nv_patch_matrix_alloc(src, PATCH_SIZE);
	nv_patch_extract(patches, src, PATCH_SIZE);
	nv_standardize_local_all(patches, 10.0f);
	nv_zca_whitening_all(patches, zca_m, 0, zca_u);
	
	conv = nv_matrix_alloc(centroids->m, GRID);
	nv_matrix_zero(conv);
	
	for (y = 0; y < patches->rows; ++y) {
		int x;
		for (x = 0; x < patches->cols; ++x) {
			nv_matrix_t *z = nv_matrix_alloc(centroids->m, 1);
			nv_matrix_t *d = nv_matrix_alloc(patches->n, 1);
			int conv_index;
			int r = (int)sqrtf(GRID);
			int x_idx = (x / (patches->cols / r));
			int y_idx = (y / (patches->rows / r));

			if (x_idx >= r) {
				x_idx = r -1;
			}
			if (y_idx >= r) {
				y_idx = r -1;
			}
			conv_index = y_idx * r + x_idx;
			if (conv_index >= GRID) {
				conv_index = GRID - 1;
			}
#if TRIANGLE_DISTANCE
			{
				float mean;
				float min_z = FLT_MAX;
				int k;
				
				for (k = 0; k < centroids->m; ++k) {
					NV_MAT_V(z, 0, k) = nv_euclidean(centroids, k, patches, NV_MAT_M(patches, y, x));
					if (NV_MAT_V(z, 0, k) < min_z) {
						min_z = NV_MAT_V(z, 0, k);
					}
				}
				mean = nv_vector_mean(z, 0);
#if TRIANGLE_DISTANCE_HALF
				mean = mean - (mean - min_z) / 4.0f;
#endif
				for (k = 0; k < centroids->m; ++k) {
					float v = mean - NV_MAT_V(z, 0, k);
					if (0.0f < v) {
#if TRIANGLE_DISTANCE_MAX						
						if (NV_MAT_V(conv, conv_index, k) < v) {
							NV_MAT_V(conv, conv_index, k) = v;
						}
#else
						NV_MAT_V(conv, conv_index, k) += v;
#endif
					}
				}
			}
#else
			{
				int nn = nv_nn(centroids, patches, NV_MAT_M(patches, y, x));
				NV_MAT_V(conv, conv_index, nn) += 1.0f;
			}
#endif
			nv_matrix_free(&z);
			nv_matrix_free(&d);
		}
	}
	for (i = 0; i < GRID; ++i) {
		memmove(&NV_MAT_V(fv, fv_j, i * conv->n),
				&NV_MAT_V(conv, i, 0), conv->n * sizeof(float));
	}
	nv_matrix_free(&patches);
	nv_matrix_free(&conv);
}
