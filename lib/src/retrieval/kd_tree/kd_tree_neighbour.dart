class KDTreeNeighbour {
  KDTreeNeighbour(this.pointIndex, this.distance);

  final int pointIndex;
  final num distance;

  @override
  bool operator ==(Object other) {
    if (other is KDTreeNeighbour) {
      return pointIndex == other.pointIndex;
    }

    return false;
  }

  @override
  String toString() => 'Distance: $distance';
}
