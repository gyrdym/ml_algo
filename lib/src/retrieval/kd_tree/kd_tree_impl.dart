import 'package:collection/collection.dart';
import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/retrieval/kd_tree/exceptions/invalid_query_point_length.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_json_keys.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_neighbour.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_node.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

part 'kd_tree_impl.g.dart';

@JsonSerializable()
class KDTreeImpl with SerializableMixin implements KDTree {
  KDTreeImpl(
      this.points, this.leafSize, this.root, this.dtype, this.schemaVersion);

  factory KDTreeImpl.fromJson(Map<String, dynamic> json) =>
      _$KDTreeImplFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$KDTreeImplToJson(this);

  @override
  @JsonKey(name: kdTreePointsJsonKey)
  final Matrix points;

  @override
  @JsonKey(name: kdTreeLeafSizeJsonKey)
  final int leafSize;

  @JsonKey(name: kdTreeRootJsonKey)
  final KDTreeNode root;

  @override
  @JsonKey(name: kdTreeDTypeJsonKey)
  final DType dtype;

  @override
  @JsonKey(name: kdTreeSchemaVersionJsonKey)
  final int schemaVersion;

  int searchIterationCount = 0;

  @override
  Iterable<KDTreeNeighbour> query(Vector point, int k,
      [Distance distanceType = Distance.euclidean]) {
    if (point.length != points.columnsNum) {
      throw InvalidQueryPointLength(point.length, points.columnsNum);
    }

    searchIterationCount = 0;

    final neighbours = HeapPriorityQueue<KDTreeNeighbour>((a, b) =>
        (point.distanceTo(points[b.index], distance: distanceType) -
                point.distanceTo(points[a.index], distance: distanceType))
            .toInt());

    _findKNNRecursively(root, point, k, neighbours, distanceType);

    return neighbours.toList().reversed;
  }

  void _findKNNRecursively(KDTreeNode? node, Vector point, int k,
      HeapPriorityQueue<KDTreeNeighbour> neighbours, Distance distanceType) {
    if (node == null) {
      return;
    }

    if (node.isLeaf) {
      _knnSearch(point, node.pointIndices, neighbours, k, distanceType);

      return;
    }

    final nodePoint = points[node.pointIndices[0]];
    final isQueueFilled = neighbours.length == k;

    if (isQueueFilled && _isNodeToFar(node, point, neighbours, distanceType)) {
      return;
    }

    _knnSearch(point, node.pointIndices, neighbours, k, distanceType);

    if (point[node.splitIndex] < nodePoint[node.splitIndex]) {
      _findKNNRecursively(node.left, point, k, neighbours, distanceType);
      _findKNNRecursively(node.right, point, k, neighbours, distanceType);
    } else {
      _findKNNRecursively(node.right, point, k, neighbours, distanceType);
      _findKNNRecursively(node.left, point, k, neighbours, distanceType);
    }
  }

  void _knnSearch(
      Vector point,
      List<int> pointIndices,
      HeapPriorityQueue<KDTreeNeighbour> neighbours,
      int k,
      Distance distanceType) {
    pointIndices.forEach((candidateIdx) {
      searchIterationCount++;
      final candidate = points[candidateIdx];
      final candidateDistance =
          candidate.distanceTo(point, distance: distanceType);
      final lastNeighbourDistance =
          neighbours.length > 0 ? neighbours.first.distance : candidateDistance;
      final isGoodCandidate = candidateDistance < lastNeighbourDistance;
      final isQueueNotFilled = neighbours.length < k;

      if (isGoodCandidate || isQueueNotFilled) {
        neighbours.add(KDTreeNeighbour(candidateIdx, candidateDistance));

        if (neighbours.length == k + 1) {
          neighbours.removeFirst();
        }
      }
    });
  }

  bool _isNodeToFar(KDTreeNode node, Vector point,
      HeapPriorityQueue<KDTreeNeighbour> neighbours, Distance distanceType) {
    if (neighbours.length == 0) {
      return false;
    }

    final nodePoint = points[node.pointIndices[0]];

    switch (distanceType) {
      case Distance.euclidean:
      case Distance.manhattan:
        return (point[node.splitIndex] - nodePoint[node.splitIndex]).abs() >
            neighbours.first.distance;

      case Distance.hamming:
      case Distance.cosine:
        final other = nodePoint.set(node.splitIndex, point[node.splitIndex]);

        return other.distanceTo(nodePoint, distance: distanceType) >
            neighbours.first.distance;

      default:
        throw UnsupportedError(
            'Distance type $distanceType is not supported yet');
    }
  }
}
