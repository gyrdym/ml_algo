import 'package:collection';
import 'package:collection/collection.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree.dart';
import 'package:ml_algo/src/retrieval/kd_tree/kd_tree_node.dart';
import 'package:ml_linalg/dtype.dart';
import 'package:ml_linalg/vector.dart';

class KDTreeImpl implements KDTree {
  KDTreeImpl(this.leafSize, this.root, this.dtype);

  @override
  final int leafSize;

  @override
  final KDTreeNode root;

  @override
  final DType dtype;

  @override
  Iterable<Vector> query(Vector sample, int k) {
    final neighbours = HeapPriorityQueue<Vector>();

    _findKNNRecursively(root, sample, k, neighbours);

    return neighbours.toList().reversed;
  }

  void _findKNNRecursively(KDTreeNode node, Vector sample, int k, HeapPriorityQueue<Vector> neighbours) {
    if (node.isLeaf) {
      // do brute force KNN on the node

      return;
    }

    final currentDistance = node.value!.distanceTo(sample);
    final lastBestDistance = neighbours.first.distanceTo(sample);

    if (currentDistance > lastBestDistance) {
      return;
    }

    neighbours.add(node.value!);

    if (sample[node.index!] < node.value![node.index!]) {
      _findKNNRecursively(node.left!, sample, k, neighbours);
      _findKNNRecursively(node.right!, sample, k, neighbours);
    } else {
      _findKNNRecursively(node.right!, sample, k, neighbours);
      _findKNNRecursively(node.left!, sample, k, neighbours);
    }
  }
}
