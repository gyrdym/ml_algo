import 'package:collection/collection.dart';
import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/retrieval/kd_tree/exceptions/invalid_query_point_length.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_json_keys.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_neighbour.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_node.dart';
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
  Iterable<KDTreeNeighbour> query(Vector point, int k) {
    if (point.length != points.columnsNum) {
      throw InvalidQueryPointLength(point.length, points.columnsNum);
    }

    searchIterationCount = 0;

    final neighbours = HeapPriorityQueue<KDTreeNeighbour>((a, b) =>
        (point.distanceTo(points[b.index]) - point.distanceTo(points[a.index]))
            .toInt());

    _findKNNRecursively(root, point, k, neighbours);

    return neighbours.toList().reversed;
  }

  void _findKNNRecursively(KDTreeNode node, Vector point, int k,
      HeapPriorityQueue<KDTreeNeighbour> neighbours) {
    searchIterationCount++;

    if (node.isLeaf) {
      _knnSearch(point, node.pointIndices, neighbours, k);

      return;
    }

    final nodePoint = points[node.pointIndices[0]];
    final boundaryPoint = _getBoundary(point, node);
    final isNodeTooFar = neighbours.length > 0 &&
        boundaryPoint.distanceTo(nodePoint) > neighbours.first.distance;
    final isQueueFilled = neighbours.length == k;

    if (isQueueFilled && isNodeTooFar) {
      return;
    }

    if (node.left == null) {
      _findKNNRecursively(node.right!, point, k, neighbours);
    } else if (node.right == null) {
      _findKNNRecursively(node.left!, point, k, neighbours);
    } else if (point[node.splitIndex!] < nodePoint[node.splitIndex!]) {
      _findKNNRecursively(node.left!, point, k, neighbours);
      _findKNNRecursively(node.right!, point, k, neighbours);
    } else {
      _findKNNRecursively(node.right!, point, k, neighbours);
      _findKNNRecursively(node.left!, point, k, neighbours);
    }
  }

  void _knnSearch(Vector point, List<int> pointIndices,
      HeapPriorityQueue<KDTreeNeighbour> neighbours, int k) {
    pointIndices.forEach((candidateIdx) {
      final candidate = points[candidateIdx];
      final candidateDistance = candidate.distanceTo(point);
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

  Vector _getBoundary(Vector point, KDTreeNode node) {
    final nodePoint = points[node.pointIndices[0]];
    final boundarySrc = [...nodePoint];

    boundarySrc[node.splitIndex!] = point[node.splitIndex!];

    return Vector.fromList(boundarySrc, dtype: point.dtype);
  }
}
