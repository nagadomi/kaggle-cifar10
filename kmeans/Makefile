validation: validation.cpp kmeans_feature.c param.h
#	g++ validation.cpp kmeans_feature.c ../lib/fileinfo.cpp -I../lib -lnv -g -o validation
	g++ validation.cpp kmeans_feature.c ../lib/fileinfo.cpp -Ofast -fopenmp -I../lib -lnv -DNDEBUG -o validation

submission: submission.cpp kmeans_feature.c param.h
	g++ submission.cpp kmeans_feature.c ../lib/fileinfo.cpp -Ofast -fopenmp -I../lib -lnv -DNDEBUG -o submission
#	g++ submission.cpp kmeans_feature.c ../lib/fileinfo.cpp -I../lib -lnv -g -o submission

centroids_learning: centroids_learning.cpp param.h
	g++ centroids_learning.cpp ../lib/fileinfo.cpp -I../lib -lnv -Ofast -fopenmp -o centroids_learning

centroids_visualization: centroids_visualization.c
	gcc centroids_visualization.c -lnv `pkg-config opencv --libs` -o centroids_visualization

clean:
	rm -f centroids_learning
	rm -f centroids_visualization
	rm -f validation
	rm -f submission
