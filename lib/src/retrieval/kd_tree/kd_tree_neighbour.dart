import 'package:ml_linalg/vector.dart';

class KDTreeNeighbour {
  KDTreeNeighbour(this.point, this.distance);

  final Vector point;
  final num distance;

  @override
  String toString() => 'Distance: $distance';
}
