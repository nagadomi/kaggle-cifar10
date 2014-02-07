#define PATCH_SIZE 6
#define IMG_SIZE 32
#define TRAIN_FILE "../data/train.txt"
#define TEST_FILE "../data/test.txt"
#define CENTROIDS 4000
#define SAMPLES (CENTROIDS * 300)
#define GRID 4
#define TRIANGLE_DISTANCE 1
#define TRIANGLE_DISTANCE_HALF 0
#define TRIANGLE_DISTANCE_MAX 0
#define DATA_N (CENTROIDS * (GRID))
#define CLASS 10
#define HIDDEN_UNIT 512

#ifdef __cplusplus
extern "C" {
#endif
void
kmeans_feature(nv_matrix_t *fv, int fv_j,
			   const nv_matrix_t *src,
			   const nv_matrix_t *zca_m,
			   const nv_matrix_t *zca_u,
			   const nv_matrix_t *centroids);

#ifdef __cplusplus
}
#endif
