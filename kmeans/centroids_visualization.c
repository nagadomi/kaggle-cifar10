#include "opencv/cxcore.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "nv_core.h"
#include "nv_num.h"
#include "nv_io.h"
#include "param.h"

#define DRAW_MAX 1000

void
save_patches(nv_matrix_t *patches, const char *filename)
{
	int max_m = NV_MIN(patches->m, DRAW_MAX);
	nv_matrix_t *image = nv_matrix3d_alloc(3, PATCH_SIZE, (PATCH_SIZE + 1) * max_m);
	nv_matrix_t *resize = nv_matrix3d_alloc(3, image->rows * 2, image->cols * 2);
	int i;
	IplImage *cv;

	nv_matrix_zero(image);
	for (i = 0; i < max_m; ++i) {
		int x, y;
		nv_vector_normalize_shift(patches, i, 0.0f, 255.0f);
		for (y = 0; y < PATCH_SIZE; ++y) {
			for (x = 0; x < PATCH_SIZE; ++x) {
				NV_MAT3D_V(image, y, (PATCH_SIZE + 1) * i + x, 0) = NV_MAT_V(patches, i, y * PATCH_SIZE * 3 + x * 3 + 0);
				NV_MAT3D_V(image, y, (PATCH_SIZE + 1) * i + x, 1) = NV_MAT_V(patches, i, y * PATCH_SIZE * 3 + x * 3 + 1);
				NV_MAT3D_V(image, y, (PATCH_SIZE + 1) * i + x, 2) = NV_MAT_V(patches, i, y * PATCH_SIZE * 3 + x * 3 + 2);
			}
		}
	}
	nv_resize(resize, image);
	cv = nv_conv_nv2ipl(resize);
	cvSaveImage(filename, cv, 0);
	
	cvReleaseImage(&cv);
	nv_matrix_free(&image);
	nv_matrix_free(&resize);
}

int main(void)
{
	nv_matrix_t *centroids = nv_load_matrix_bin("centroids.mat");
	save_patches(centroids, "centroids.png");
	nv_matrix_free(&centroids);

	return 0;
}
