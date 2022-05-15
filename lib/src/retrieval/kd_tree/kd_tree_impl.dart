import 'package:collection/collection.dart';
import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/retrieval/kd_tree/exceptions/invalid_query_point_length.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_json_keys.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_node.dart';
import 'package:ml_algo/src/retrieval/mixins/knn_searcher.dart';
import 'package:ml_algo/src/retrieval/neighbour.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

part 'kd_tree_impl.g.dart';

@JsonSerializable()
class KDTreeImpl with SerializableMixin, KnnSearcherMixin implements KDTree {
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

  @override
  Iterable<Neighbour> query(Vector point, int k,
      [Distance distanceType = Distance.euclidean]) {
    if (point.length != points.columnsNum) {
      throw InvalidQueryPointLength(point.length, points.columnsNum);
    }

    final neighbours = createQueue(point, distanceType);

    _findKNNRecursively(root, point, k, neighbours, distanceType);

    return neighbours.toList().reversed;
  }

  @override
  Iterable<Neighbour> queryIterable(Iterable<num> point, int k,
          [Distance distanceType = Distance.euclidean]) =>
      query(Vector.fromList(point.toList(), dtype: dtype), k);

  void _findKNNRecursively(KDTreeNode? node, Vector point, int k,
      HeapPriorityQueue<Neighbour> neighbours, Distance distanceType) {
    if (node == null) {
      return;
    }

    if (node.isLeaf) {
      search(point, node.pointIndices, neighbours, k, distanceType);

      return;
    }

    final nodePoint = points[node.pointIndices[0]];
    final isQueueFilled = neighbours.length == k;

    if (isQueueFilled && _isNodeToFar(node, point, neighbours, distanceType)) {
      return;
    }

    search(point, node.pointIndices, neighbours, k, distanceType);

    if (point[node.splitIndex] < nodePoint[node.splitIndex]) {
      _findKNNRecursively(node.left, point, k, neighbours, distanceType);
      _findKNNRecursively(node.right, point, k, neighbours, distanceType);
    } else {
      _findKNNRecursively(node.right, point, k, neighbours, distanceType);
      _findKNNRecursively(node.left, point, k, neighbours, distanceType);
    }
  }

  bool _isNodeToFar(KDTreeNode node, Vector point,
      HeapPriorityQueue<Neighbour> neighbours, Distance distanceType) {
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
