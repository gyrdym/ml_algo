import 'package:ml_algo/src/model_selection/split_indices_provider/lpo_indices_provider.dart';
import 'package:ml_algo/src/retrieval/mixins/knn_searcher.dart';
import 'package:ml_algo/src/retrieval/neighbour.dart';
import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/helpers/get_binary_representation.dart';
import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/helpers/get_indices_from_binary_representation.dart';
import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/helpers/group_indices_by_bins.dart';
import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/random_binary_projection_searcher.dart';
import 'package:ml_dataframe/ml_dataframe.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

class RandomBinaryProjectionSearcherImpl
    with KnnSearcherMixin
    implements RandomBinaryProjectionSearcher {
  RandomBinaryProjectionSearcherImpl(DataFrame dataset, this.digitCapacity,
      {int seed = 0, DType dtype = DType.float32})
      : header = dataset.header,
        points = dataset.toMatrix(dtype),
        dtype = dtype {
    randomVectors = Matrix.random(points.columnsNum, digitCapacity, seed: seed);
    bins = groupIndicesByBins(getBinIdsFromBinaryRepresentation(
        getBinaryRepresentation(points, randomVectors)));
  }

  final int digitCapacity;
  final Iterable<String> header;
  final DType dtype;

  @override
  final Matrix points;

  late Matrix randomVectors;
  late Map<num, List<int>> bins;

  @override
  Iterable<Neighbour> query(Vector point, int k, int searchRadius) {
    final pointAsMatrix = Matrix.fromRows([point], dtype: dtype);
    final queryBits =
        getBinaryRepresentation(pointAsMatrix, randomVectors).toVector();
    final candidateIndices = <int>[];

    for (var i = 0; i < searchRadius; i++) {
      final indicesProvider = LpoIndicesProvider(i);
      final indexGroups = indicesProvider.getIndices(randomVectors.columnsNum);

      for (final indices in indexGroups) {
        final queryBitsFlipped = queryBits.toList();

        for (final index in indices) {
          queryBitsFlipped[index] = queryBitsFlipped[index] == 1 ? 0 : 1;
        }

        final flippedBitsAsMatrix = Matrix.fromList([queryBitsFlipped]);
        final nearbyBinId =
            getBinIdsFromBinaryRepresentation(flippedBitsAsMatrix)[0];

        if (bins.containsKey(nearbyBinId)) {
          candidateIndices.addAll(bins[nearbyBinId]!);
        }
      }
    }

    final queue = createQueue(point, Distance.euclidean);

    search(point, candidateIndices, queue, k, Distance.euclidean);

    return queue.toList().reversed;
  }
}
