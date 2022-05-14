import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/common/constants/common_json_keys.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/model_selection/split_indices_provider/lpo_indices_provider.dart';
import 'package:ml_algo/src/retrieval/mixins/knn_searcher.dart';
import 'package:ml_algo/src/retrieval/neighbour.dart';
import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/helpers/get_binary_representation.dart';
import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/helpers/get_indices_from_binary_representation.dart';
import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/helpers/group_indices_by_bins.dart';
import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/random_binary_projection_searcher.dart';
import 'package:ml_algo/src/retrieval/random_binary_projection_searcher/random_binary_projection_searcher_json_keys.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

part 'random_binary_projection_searcher_impl.g.dart';

@JsonSerializable()
class RandomBinaryProjectionSearcherImpl
    with SerializableMixin, KnnSearcherMixin
    implements RandomBinaryProjectionSearcher {
  RandomBinaryProjectionSearcherImpl(
      this.header, this.points, this.digitCapacity,
      {this.seed, this.schemaVersion = 1}) {
    randomVectors = Matrix.random(points.columnsNum, digitCapacity,
        seed: seed, dtype: points.dtype);
    bins = groupIndicesByBins(getBinIdsFromBinaryRepresentation(
        getBinaryRepresentation(points, randomVectors)));
  }

  factory RandomBinaryProjectionSearcherImpl.fromJson(
          Map<String, dynamic> json) =>
      _$RandomBinaryProjectionSearcherImplFromJson(json);

  @override
  Map<String, dynamic> toJson() =>
      _$RandomBinaryProjectionSearcherImplToJson(this);

  @override
  @JsonKey(name: randomBinaryProjectionSeedJsonKey)
  final int? seed;

  @override
  @JsonKey(name: randomBinaryProjectionDigitCapacityJsonKey)
  final int digitCapacity;

  @override
  @JsonKey(name: randomBinaryProjectionHeaderJsonKey)
  final Iterable<String> header;

  @override
  @JsonKey(name: randomBinaryProjectionPointsJsonKey)
  final Matrix points;

  @JsonKey(name: randomBinaryProjectionRandomVectorsJsonKey)
  late Matrix randomVectors;

  @JsonKey(name: randomBinaryProjectionBinsJsonKey)
  late Map<int, List<int>> bins;

  @override
  @JsonKey(name: jsonSchemaVersionJsonKey)
  final int schemaVersion;

  @override
  Iterable<Neighbour> query(Vector point, int k, int searchRadius) {
    final pointAsMatrix = Matrix.fromRows([point], dtype: points.dtype);
    final queryBits =
        getBinaryRepresentation(pointAsMatrix, randomVectors).toVector();
    final candidateIndices = <int>[];

    for (var i = 0; i < searchRadius + 1; i++) {
      final indicesProvider = LpoIndicesProvider(i);
      final indexGroups = indicesProvider.getIndices(randomVectors.columnsNum);

      for (final indices in indexGroups) {
        final queryBitsFlipped = queryBits.toList();

        for (final index in indices) {
          queryBitsFlipped[index] = queryBitsFlipped[index] == 1 ? 0 : 1;
        }

        final flippedBitsAsMatrix =
            Matrix.fromList([queryBitsFlipped], dtype: points.dtype);
        final nearbyBinId =
            getBinIdsFromBinaryRepresentation(flippedBitsAsMatrix).first;

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
