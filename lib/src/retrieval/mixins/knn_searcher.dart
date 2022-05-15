import 'package:collection/collection.dart';
import 'package:ml_algo/src/retrieval/neighbour.dart';
import 'package:ml_linalg/distance.dart';
import 'package:ml_linalg/matrix.dart';
import 'package:ml_linalg/vector.dart';

mixin KnnSearcherMixin {
  Matrix get points;

  int searchIterationCount = 0;

  HeapPriorityQueue<Neighbour> createQueue(
      Vector point, Distance distanceType) {
    searchIterationCount = 0;

    return HeapPriorityQueue<Neighbour>((a, b) {
      final distanceA =
          point.distanceTo(points[a.index], distance: distanceType);
      final distanceB =
          point.distanceTo(points[b.index], distance: distanceType);

      if (distanceA < distanceB) {
        return 1;
      }

      if (distanceA > distanceB) {
        return -1;
      }

      return 0;
    });
  }

  void search(Vector point, List<int> pointIndices,
      HeapPriorityQueue<Neighbour> neighbours, int k, Distance distanceType) {
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
        neighbours.add(Neighbour(candidateIdx, candidateDistance));

        if (neighbours.length == k + 1) {
          neighbours.removeFirst();
        }
      }
    });
  }
}
