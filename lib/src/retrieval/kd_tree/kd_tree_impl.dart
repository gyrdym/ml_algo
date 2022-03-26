import 'package:collection/collection.dart';
import 'package:json_annotation/json_annotation.dart';
import 'package:ml_algo/src/common/serializable/serializable_mixin.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_json_keys.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_neighbour.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_node.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';

part 'kd_tree_impl.g.dart';

@JsonSerializable()
class KDTreeImpl with SerializableMixin implements KDTree {
  KDTreeImpl(this.leafSize, this.root, this.dtype);

  factory KDTreeImpl.fromJson(Map<String, dynamic> json) =>
      _$KDTreeImplFromJson(json);

  @override
  Map<String, dynamic> toJson() => _$KDTreeImplToJson(this);

  @override
  @JsonKey(name: kdTreeLeafSizeJsonKey)
  final int leafSize;

  @override
  @JsonKey(name: kdTreeRootJsonKey)
  final KDTreeNode root;

  @override
  @JsonKey(name: kdTreeDTypeJsonKey)
  final DType dtype;

  int searchIterationCount = 0;

  @override
  Iterable<KDTreeNeighbour> query(Vector point, int k) {
    searchIterationCount = 0;

    final neighbours = HeapPriorityQueue<KDTreeNeighbour>(
        (a, b) => (point.distanceTo(b.point) - point.distanceTo(a.point)).toInt());

    _findKNNRecursively(root, point, k, neighbours);

    return neighbours.toList().reversed;
  }

  void _findKNNRecursively(KDTreeNode node, Vector point, int k,
      HeapPriorityQueue<KDTreeNeighbour> neighbours) {
    searchIterationCount++;

    if (node.isLeaf) {
      node.points!.rows.forEach((vector) {
        _knnSearch(point, vector, neighbours, k);
      });

      return;
    }

    if (neighbours.length == k &&
        point.distanceTo(node.value!) > neighbours.first.distance) {
      return;
    }

    _knnSearch(point, node.value!, neighbours, k);

    if (node.left == null) {
      _findKNNRecursively(node.right!, point, k, neighbours);
    } else if (node.right == null) {
      _findKNNRecursively(node.left!, point, k, neighbours);
    } else if (point[node.splitIndex!] < node.value![node.splitIndex!]) {
      _findKNNRecursively(node.left!, point, k, neighbours);
      _findKNNRecursively(node.right!, point, k, neighbours);
    } else {
      _findKNNRecursively(node.left!, point, k, neighbours);
      _findKNNRecursively(node.right!, point, k, neighbours);
    }
  }

  void _knnSearch(Vector point, Vector neighbourCandidate,
      HeapPriorityQueue<KDTreeNeighbour> neighbours, int k) {
    final candidateDistance = neighbourCandidate.distanceTo(point);
    final lastNeighbourDistance = neighbours.length > 0
        ? neighbours.first.distance
        : candidateDistance;

    if (candidateDistance < lastNeighbourDistance || neighbours.length < k) {
      neighbours.add(KDTreeNeighbour(neighbourCandidate, candidateDistance));

      if (neighbours.length == k + 1) {
        neighbours.removeFirst();
      }
    }
  }

  @override
  // TODO: implement schemaVersion
  int? get schemaVersion => throw UnimplementedError();
}
